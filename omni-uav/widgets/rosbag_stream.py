from pathlib import Path
from typing import Optional, List
import numpy as np
from PyQt5 import QtGui

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from rosbags.image import message_to_cvimage
    ROSBAG_AVAILABLE = True
except ImportError:
    ROSBAG_AVAILABLE = False
    print("Warning: rosbags and rosbags-image not available. Install with: pip install rosbags rosbags-image")



class RosbagStream:
    """Stream images from a ROS bag file.

    Args:
        bag_path: Path to the .bag file
        topic: ROS topic name for this camera (e.g., "/airsim_node/drone1/front_center_custom/Scene")
        camera_id: Identifier for this camera stream
    """

    def __init__(self, bag_path: Path, topic: str, camera_id: str):
        if not ROSBAG_AVAILABLE:
            raise ImportError("rosbags and rosbags-image are required. Install with: pip install rosbags rosbags-image")

        self.bag_path = bag_path
        self.topic = topic
        self.camera_id = camera_id
        self.last_image: Optional[QtGui.QImage] = None

        # Load all messages from this topic into memory for fast access
        self.messages: List = []
        self.frame_index = 0

        print(f"Loading rosbag: {bag_path} for topic: {topic}")
        try:
            # Use AnyReader which handles both ROS1 and ROS2 bags automatically
            # Determine typestore based on file extension
            if bag_path.suffix == '.bag':
                typestore = get_typestore(Stores.ROS1_NOETIC)
            else:
                typestore = get_typestore(Stores.ROS2_HUMBLE)

            with AnyReader([bag_path], default_typestore=typestore) as reader:
                # Get all messages from this topic
                connections = [conn for conn in reader.connections if conn.topic == topic]
                if not connections:
                    print(f"Warning: Topic {topic} not found in bag")
                    print(f"Available topics in bag:")
                    all_topics = set(conn.topic for conn in reader.connections)
                    for topic_name in sorted(all_topics):
                        print(f"  - {topic_name}")
                    raise ValueError(f"Topic {topic} not found in bag file")

                # Read all messages from the topic
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    # AnyReader.deserialize handles the format automatically
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    self.messages.append(msg)

            print(f"Loaded {len(self.messages)} frames from topic {topic}")

            if len(self.messages) == 0:
                print(f"Warning: No messages found for topic {topic}")
        except Exception as e:
            print(f"Error loading rosbag: {e}")
            import traceback
            traceback.print_exc()
            raise

    def read(self) -> Optional[QtGui.QImage]:
        """Read the next frame from the bag."""
        if not self.messages:
            return None

        # Get current message
        msg = self.messages[self.frame_index]

        try:
            # Convert ROS Image message to OpenCV format using rosbags-image
            # message_to_cvimage returns RGB by default, but we can specify 'bgr8' if needed
            cv_image = message_to_cvimage(msg, 'bgr8')

            # Convert BGR to RGB
            import cv2
            frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Convert to QImage
            height, width, channels = frame_rgb.shape
            bytes_per_line = channels * width
            image = QtGui.QImage(
                frame_rgb.data,
                width,
                height,
                bytes_per_line,
                QtGui.QImage.Format_RGB888
            ).copy()

            self.last_image = image

            # Advance to next frame (loop back to start)
            self.frame_index = (self.frame_index + 1) % len(self.messages)

            return image

        except Exception as e:
            print(f"Error converting ROS message to image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_latest(self) -> Optional[QtGui.QImage]:
        """Get the most recently read frame."""
        return self.last_image

    def close(self):
        """Clean up resources."""
        self.messages.clear()
