# ROS Bag Support

To use rosbag files with OmniUAV, you need to install the required dependencies:

```bash
pip install rosbags rosbags-image
```

**Note**: These are pure Python packages that work without requiring ROS installation. They support both ROS1 (`.bag`) and ROS2 (`.db3`) bag file formats.

## Usage

Place your `.bag` file in the data directory (e.g., `examples/multi_drone_images.bag`).

The application will automatically detect and load the bag file with the following topic mapping:

- **Drone 1**: `/airsim_node/drone1/front_center_custom/Scene` → 无人机-01
- **Drone 2**: `/airsim_node/drone2/front_center_custom/Scene` → 无人机-02
- **Drone 3**: `/airsim_node/drone3/front_center_custom/Scene` → 无人机-03
- **Drone 4**: `/airsim_node/drone4/front_center_custom/Scene` → 无人机-04

## Priority Order

The application checks for data sources in this order:

1. **ROS bag files** (`.bag`)
2. **Image sequences** (`rgb/` directory with `.jpg`, `.jpeg`, or `.png` files)
3. **MP4 videos** (`cam01.mp4`, `cam02.mp4`, etc.)

## Custom Topics

To use different topic names, modify the `topic_mapping` dictionary in `tabs/camera_tab.py:_init_video_streams()`.
