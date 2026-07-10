from typing import Optional, TYPE_CHECKING, List, Dict
import threading
import time
from PyQt5 import QtGui, QtCore
import cv2
import numpy as np

if TYPE_CHECKING:
    from sensor_msgs.msg import Image as ROSImage

try:
    import rospy
    from sensor_msgs.msg import Image as ROSImage
    ROS_AVAILABLE = True
    # Don't use cv_bridge due to libffi compatibility issues
    # We'll convert images manually using numpy
    print("Using manual ROS image conversion (cv_bridge bypassed due to libffi issues)")
except ImportError:
    ROS_AVAILABLE = False
    ROSImage = None  # type: ignore
    print("Warning: rospy not available. Install with:")
    print("  sudo apt-get install ros-noetic-cv-bridge ros-noetic-image-transport")
    print("  pip install rospy")


class ROSLiveStream:
    """Live ROS image topic subscriber.

    Subscribes to a ROS image topic and provides the latest frame.
    Runs the ROS spinner in a background thread.

    Args:
        topic: ROS topic name (e.g., "/airsim_node/drone1/front_center_custom/Scene")
        camera_id: Identifier for this camera stream
        queue_size: ROS subscriber queue size (default: 1 for latest only)
    """

    def __init__(self, topic: str, camera_id: str, queue_size: int = 1):
        if not ROS_AVAILABLE:
            raise ImportError(
                "ROS Python packages (rospy, cv_bridge) are required.\n"
                "Install with: sudo apt-get install ros-noetic-cv-bridge"
            )

        self.topic = topic
        self.camera_id = camera_id
        self.last_image: Optional[QtGui.QImage] = None
        self.last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._spinner_thread: Optional[threading.Thread] = None

        # Track timestamps for frame rate
        self._last_timestamp = None
        self._frame_count = 0

        # Initialize ROS node if not already initialized
        # rospy doesn't have is_initialized(), so we check by trying to get the node name
        try:
            node_name = rospy.get_name()
            if node_name == '/unnamed':
                rospy.init_node('omni_uav_ros_stream', anonymous=True, disable_signals=True)
        except Exception:
            rospy.init_node('omni_uav_ros_stream', anonymous=True, disable_signals=True)

        # Subscribe to the image topic
        self.subscriber = rospy.Subscriber(
            topic,
            ROSImage,
            self._image_callback,
            queue_size=queue_size,
            buff_size=2**24  # Larger buffer for high-resolution images
        )

        print(f"Subscribed to ROS topic: {topic}")

        # Start spinner in background thread
        self._start_spinner()

    def _image_callback(self, msg):  # type: ignore
        """ROS image callback - stores the latest frame."""
        try:
            # Convert ROS Image to numpy array manually (bypassing cv_bridge)
            # Get image dimensions
            height = msg.height
            width = msg.width

            # Get the raw image data
            if msg.encoding == "bgr8":
                # BGR format, 8 bits per channel
                dtype = np.uint8
                channels = 3
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, channels)
                # Convert BGR to RGB (OpenCV uses BGR by default)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            elif msg.encoding == "rgb8":
                # RGB format, 8 bits per channel
                dtype = np.uint8
                channels = 3
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, channels)
            elif msg.encoding == "mono8":
                # Grayscale, 8 bits
                dtype = np.uint8
                channels = 1
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(height, width)
                # Convert grayscale to RGB for display
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            elif msg.encoding == "bayer_rggb8":
                # Bayer pattern - need to convert
                dtype = np.uint8
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(height, width)
                # Use OpenCV to demosaic
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BayerBG2RGB)
            else:
                print(f"[{self.topic}] Unsupported encoding: {msg.encoding}, attempting direct conversion")
                # Try direct conversion as fallback
                dtype = np.uint8
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, -1)
                if len(cv_image.shape) == 2:  # (H, W)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                elif cv_image.shape[2] == 4:  # BGRA
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)

            with self._lock:
                self.last_frame = cv_image
                self._frame_count += 1
                self._last_timestamp = time.time()

            # Log first frame received
            if self._frame_count == 1:
                print(f"[{self.topic}] First frame received: {cv_image.shape}, encoding={msg.encoding}")

        except Exception as e:
            print(f"[{self.topic}] Error in ROS image callback: {e}")
            import traceback
            traceback.print_exc()

    def _spin_loop(self):
        """Background thread that spins ROS."""
        while self._running and not rospy.is_shutdown():
            try:
                rospy.spin_once(timeout_sec=0.1)
            except Exception:
                break

    def _start_spinner(self):
        """Start the ROS spinner in a background thread."""
        if self._running:
            return

        self._running = True
        self._spinner_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spinner_thread.start()
        print(f"ROS spinner started for {self.topic}")

        # Warn if no frames received after 5 seconds
        def _check_frames():
            time.sleep(5)
            with self._lock:
                if self._frame_count == 0:
                    print(f"[{self.topic}] Warning: No frames received after 5 seconds. Check:")
                    print(f"  - Topic name: {self.topic}")
                    print(f"  - Run: rostopic echo {self.topic}")
                    print(f"  - Run: rostopic list")

        threading.Thread(target=_check_frames, daemon=True).start()

    def read(self) -> Optional[QtGui.QImage]:
        """Get the latest frame from ROS topic.

        Returns:
            QImage if a frame is available (even if it's an old frame), None otherwise
        """
        with self._lock:
            if self.last_frame is None:
                # No frame received yet
                return None

            # last_frame is already in RGB format (converted in callback)
            frame_rgb = self.last_frame
            height, width = frame_rgb.shape[0], frame_rgb.shape[1]

            # Create QImage from numpy array
            image = QtGui.QImage(
                frame_rgb.data,
                width,
                height,
                frame_rgb.strides[0],
                QtGui.QImage.Format_RGB888,
            ).copy()

            self.last_image = image

            # Debug: print every 10th frame to reduce spam (disabled - too verbose)
            # if self._frame_count % 10 == 0:
            #     print(f"[{self.topic}] Frame #{self._frame_count}: {width}x{height}")

            return image

    def get_latest(self) -> Optional[QtGui.QImage]:
        """Get the most recently read frame."""
        return self.last_image

    def is_alive(self) -> bool:
        """Check if the ROS connection is still active."""
        return self._running and not rospy.is_shutdown()

    def get_fps(self) -> float:
        """Get the current frame rate from ROS topic."""
        with self._lock:
            if self._last_timestamp is None or self._frame_count < 2:
                return 0.0

            # Simple FPS calculation based on frame count
            # For more accurate FPS, you'd need to track individual timestamps
            elapsed = time.time() - (self._start_time if hasattr(self, '_start_time') else self._last_timestamp)
            if not hasattr(self, '_start_time'):
                self._start_time = self._last_timestamp
                elapsed = 1.0

            return self._frame_count / max(elapsed, 0.001)

    def get_frame_count(self) -> int:
        """Get the total number of frames received."""
        with self._lock:
            return self._frame_count

    def resubscribe(self, new_topic: str) -> bool:
        """Resubscribe to a different ROS topic.

        Args:
            new_topic: New ROS topic name to subscribe to

        Returns:
            True if resubscription was successful, False otherwise
        """
        try:
            # Unregister old subscriber
            if self.subscriber:
                self.subscriber.unregister()

            # Update topic
            old_topic = self.topic
            self.topic = new_topic

            # Reset frame tracking
            with self._lock:
                self._frame_count = 0
                self._last_timestamp = None
                self.last_frame = None
                self.last_image = None

            # Create new subscriber
            self.subscriber = rospy.Subscriber(
                new_topic,
                ROSImage,
                self._image_callback,
                queue_size=1,
                buff_size=2**24
            )

            print(f"[{old_topic}] Resubscribed to: {new_topic}")
            return True

        except Exception as e:
            print(f"[{self.topic}] Failed to resubscribe to {new_topic}: {e}")
            return False

    def has_frames(self) -> bool:
        """Check if any frames have been received."""
        with self._lock:
            return self._frame_count > 0

    def close(self):
        """Clean up ROS resources."""
        self._running = False

        if self.subscriber:
            self.subscriber.unregister()

        if self._spinner_thread:
            self._spinner_thread.join(timeout=1.0)

        with self._lock:
            self.last_frame = None
            self.last_image = None

        print(f"ROS stream closed for {self.topic}")


class TrackCropsSubscriber(QtCore.QObject):
    """Subscriber for ROS track crops topics.

    Subscribes to a track_crops topic that contains image crops of tracked objects.
    Stores the latest crops for LLM analysis.

    Args:
        topic: ROS topic name (e.g., "/airsim_node/drone1/front_center_custom/Scene/track_crops")
        drone_id: Drone identifier (e.g., "drone1", "drone2")
    """

    # Signal emitted when crops are received (drone_id, crop_count)
    crops_received = QtCore.pyqtSignal(str, int)

    def __init__(self, topic: str, drone_id: str):
        super().__init__()
        if not ROS_AVAILABLE:
            raise ImportError("ROS Python packages are required.")

        self.topic = topic
        self.drone_id = drone_id
        self._lock = threading.Lock()
        self._running = False
        self._spinner_thread: Optional[threading.Thread] = None

        # Store the latest track crops data
        # Format: list of dicts with 'image' (numpy array) and 'id' (tracker ID)
        self.track_crops: List[Dict] = []
        self._last_crops_timestamp = None

        # Track if we've already notified about crops (to avoid spam)
        self._notified_crops = False

        # Initialize ROS node if needed
        try:
            node_name = rospy.get_name()
            if node_name == '/unnamed':
                rospy.init_node('omni_uav_track_crops', anonymous=True, disable_signals=True)
        except Exception:
            rospy.init_node('omni_uav_track_crops', anonymous=True, disable_signals=True)

        # Subscribe to track_crops topic
        # Assuming the message type is sensor_msgs/Image for crops
        self.subscriber = rospy.Subscriber(
            topic,
            ROSImage,
            self._track_crops_callback,
            queue_size=10,
            buff_size=2**24
        )

        # print(f"[TrackCrops] Subscribed to {topic} for {drone_id}")

        # Start spinner in background thread
        self._start_spinner()

    def _track_crops_callback(self, msg):
        """Callback for track_crops images.

        Note: This assumes the track_crops topic publishes individual crop images.
        In practice, you might need a custom message format that includes:
        - The crop image
        - The tracker ID
        - Metadata (timestamp, drone_id, etc.)
        """
        try:
            # Convert ROS Image to numpy array
            height = msg.height
            width = msg.width

            if msg.encoding == "bgr8":
                dtype = np.uint8
                channels = 3
                crop_image = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, channels)
                crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            elif msg.encoding == "rgb8":
                dtype = np.uint8
                channels = 3
                crop_image = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, channels)
            else:
                # Default handling
                dtype = np.uint8
                crop_image = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, -1)
                if len(crop_image.shape) == 2:
                    crop_image = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2RGB)
                elif crop_image.shape[2] == 4:
                    crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGRA2RGB)

            # For this implementation, we'll store crops in a list
            # In a real system, you'd want to associate each crop with its tracker ID
            # This might require a custom ROS message format

            with self._lock:
                # Extract frame_id from ROS message header
                frame_id = msg.header.frame_id if hasattr(msg, 'header') else f"frame_{int(time.time() * 1000)}"

                # Store crop with frame_id for grouping
                self.track_crops.append({
                    'image': crop_image,
                    'frame_id': frame_id,
                    'timestamp': time.time()
                })
                self._last_crops_timestamp = time.time()

                # Print when we receive a crop (for debugging) - disabled
                # print(f"[TrackCrops] Received crop from {self.drone_id}: ID={tracker_id}, size={width}x{height}")

                # Emit signal to notify UI (only once when we first receive crops)
                if not self._notified_crops:
                    self._notified_crops = True
                    self.crops_received.emit(self.drone_id, len(self.track_crops))

                # Keep only the most recent 50 crops to avoid memory issues
                if len(self.track_crops) > 50:
                    self.track_crops = self.track_crops[-50:]

        except Exception as e:
            # print(f"[TrackCrops] Error in callback: {e}")
            import traceback
            traceback.print_exc()

    def _spin_loop(self):
        """Background thread that spins ROS."""
        while self._running and not rospy.is_shutdown():
            try:
                rospy.spin_once(timeout_sec=0.1)
            except Exception:
                break

    def _start_spinner(self):
        """Start the ROS spinner in a background thread."""
        if self._running:
            return

        self._running = True
        self._spinner_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spinner_thread.start()
        # print(f"[TrackCrops] Spinner started for {self.topic}")

    def get_latest_crops(self) -> List[Dict]:
        """Get the latest track crops.

        Returns:
            List of dicts with 'image' (numpy array) and 'frame_id'
        """
        with self._lock:
            return self.track_crops.copy()

    def get_latest_frame_crops(self) -> List[Dict]:
        """Get all crops from the latest frame (same frame_id).

        Returns:
            List of dicts with 'image' (numpy array) and 'frame_id'
        """
        with self._lock:
            if not self.track_crops:
                return []

            # Get the frame_id of the most recent crop
            latest_frame_id = self.track_crops[-1]['frame_id']

            # Find all crops with the same frame_id (iterate backwards to get latest first)
            latest_crops = []
            for crop in reversed(self.track_crops):
                if crop['frame_id'] == latest_frame_id:
                    latest_crops.append(crop)
                else:
                    # Since crops are ordered by time, we can stop at first different frame_id
                    break

            # Reverse to maintain original order
            return list(reversed(latest_crops))

    def clear_crops(self):
        """Clear stored crops."""
        with self._lock:
            self.track_crops.clear()

    def has_crops(self) -> bool:
        """Check if any crops have been received."""
        with self._lock:
            return len(self.track_crops) > 0

    def is_alive(self) -> bool:
        """Check if the subscriber is still active."""
        return self._running and not rospy.is_shutdown()

    def close(self):
        """Clean up ROS resources."""
        self._running = False

        if self.subscriber:
            self.subscriber.unregister()

        if self._spinner_thread:
            self._spinner_thread.join(timeout=1.0)

        with self._lock:
            self.track_crops.clear()

        # print(f"[TrackCrops] Closed subscriber for {self.topic}")


class ROSTrackingPublisher:
    """Publisher for ROS tracking results.

    Publishes LLM analysis results to the llm_track_result topic.

    Args:
        drone_id: The drone ID (e.g., "drone1", "drone2")
    """

    def __init__(self, drone_id: str):
        if not ROS_AVAILABLE:
            raise ImportError("ROS Python packages are required.")

        self.drone_id = drone_id
        self._running = False

        # Extract drone number (e.g., "drone1" -> 1)
        try:
            self.drone_num = int(drone_id.replace("drone", ""))
        except ValueError:
            self.drone_num = 1

        # Build topic name: /airsim_node/drone1/front_center_custom/Scene/llm_track_result
        topic = f"/airsim_node/{drone_id}/front_center_custom/Scene/llm_track_result"

        # Initialize ROS node if needed
        try:
            node_name = rospy.get_name()
            if node_name == '/unnamed':
                rospy.init_node('omni_uav_tracking_result', anonymous=True, disable_signals=True)
        except Exception:
            rospy.init_node('omni_uav_tracking_result', anonymous=True, disable_signals=True)

        # Create publisher for String messages (JSON format)
        try:
            from std_msgs.msg import String
            self.publisher = rospy.Publisher(topic, String, queue_size=10)
            self.StringMsg = String
        except ImportError:
            self.publisher = None
            self.StringMsg = None

    def publish_result(self, is_matched: bool, matched_ids: List[int]):
        """Publish tracking result to ROS topic.

        Args:
            is_matched: Whether any objects matched the description
            matched_ids: List of tracker IDs that matched
        """
        print(f"[PUBLISH] Publishing to ROS: drone={self.drone_id}, is_matched={is_matched}, ids={matched_ids}")

        if self.publisher is None:
            print(f"[PUBLISH] ERROR: publisher is None!")
            return

        import json

        result = {
            "drone_id": self.drone_num,
            "is_matched": is_matched,
            "matched_ids": matched_ids
        }

        json_str = json.dumps(result)
        msg = self.StringMsg()
        msg.data = json_str

        self.publisher.publish(msg)
        print(f"[PUBLISH] Published: {json_str}")

    def is_alive(self) -> bool:
        """Check if the publisher is still active."""
        return self._running and not rospy.is_shutdown()

    def close(self):
        """Clean up ROS resources."""
        self._running = False

        if self.publisher:
            self.publisher.unregister()

        # print(f"[ROSTracking] Closed publisher for {self.topic}")


def test_ros_stream():
    """Test function to verify ROS live stream works."""
    import sys

    if not ROS_AVAILABLE:
        print("ROS not available. Cannot test.")
        return

    # Test topic
    topic = "/camera/image_raw"

    print(f"Testing ROS live stream on topic: {topic}")
    print("Make sure ROS is running and publishing to this topic.")
    print("Press Ctrl+C to stop.")

    try:
        stream = ROSLiveStream(topic, "test")

        # Try to get a few frames
        for i in range(30):
            time.sleep(0.1)
            image = stream.read()
            if image:
                print(f"Got frame {i+1}: {image.width()}x{image.height()}")
            else:
                print(f"No frame yet {i+1}...")

        stream.close()
        print("Test completed successfully!")

    except KeyboardInterrupt:
        stream.close()
        print("\nTest interrupted.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ros_stream()
