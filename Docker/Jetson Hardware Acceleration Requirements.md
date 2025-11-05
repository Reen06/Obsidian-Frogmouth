# Jetson Hardware Acceleration Requirements

This document outlines the extra Docker add-ons and configurations required to utilize hardware acceleration on Jetson boards versus using a standard Docker container.

## **1. NVIDIA Container Runtime**

The main requirement is the **NVIDIA Container Runtime** (not just standard Docker). This enables GPU access inside containers.

### Configuration in `/etc/docker/daemon.json`:

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

### Restart Docker Service:

```bash
sudo systemctl restart docker
```

### Verify Configuration:

```bash
sudo docker info | grep 'Default Runtime'
# Should output: Default Runtime: nvidia
```

---

## **2. Runtime Flags**

When running containers, you must use `--runtime nvidia` (or `--gpus all` on non-Jetson systems):

### For Jetson:

```bash
docker run --runtime nvidia --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
```

### For Non-Jetson Systems:

```bash
docker run --gpus all --env NVIDIA_DRIVER_CAPABILITIES=all ...
```

---

## **3. Environment Variables**

### NVIDIA_DRIVER_CAPABILITIES

- `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` - Enables GPU compute, utility functions, and graphics capabilities

---

## **4. Jetson-Specific Device Mounts**

From the `jetson-containers` project's `run.sh` script, these device mounts are required for Jetson hardware:

### Device Mounts:

- `/dev/bus/usb` - USB devices
- `/dev/snd` - Audio devices  
- V4L2 devices (`/dev/video*`) - Video cameras (including CSI cameras)
- I2C devices (`/dev/i2c-*`) - I2C buses
- `/dev/ttyACM*` - Serial devices

### Volume Mounts:

- `/tmp/argus_socket:/tmp/argus_socket` - Camera daemon socket
- `/etc/enctune.conf:/etc/enctune.conf` - Encoder tuning configuration
- `/etc/nv_tegra_release:/etc/nv_tegra_release` - Jetson version information
- `/tmp/nv_jetson_model:/tmp/nv_jetson_model` - Jetson board model
- `/var/run/dbus:/var/run/dbus` - D-Bus socket
- `/var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket` - Network discovery
- `/var/run/docker.sock:/var/run/docker.sock` - Docker socket (for nested containers)

### Other Important Settings:

- `--network host` - Host networking mode
- `--shm-size=8g` - Shared memory size for CUDA operations
- Display/X11 mounts (if `DISPLAY` environment variable is set)
- PulseAudio mounts (if available)

---

## **5. Additional Hardware-Specific Features**

The `jetson-containers` project's `run.sh` script also supports:

- **CSI Camera Conversion** - Converts CSI cameras to V4L2 webcam devices using `v4l2loopback` and GStreamer
- **Display Support** - X11 forwarding for GUI applications
- **Audio Support** - PulseAudio integration

---

## **6. Complete Example Command**

### Using jetson-containers helper (recommended):

```bash
jetson-containers run $(autotag l4t-pytorch)
```

### Manual Docker run command:

```bash
sudo docker run --runtime nvidia \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -it --rm --network host \
  --shm-size=8g \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  --volume /etc/enctune.conf:/etc/enctune.conf \
  --volume /etc/nv_tegra_release:/etc/nv_tegra_release \
  --volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \
  --volume /var/run/dbus:/var/run/dbus \
  --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --device /dev/snd \
  --device /dev/bus/usb \
  CONTAINER:TAG
```

---

## **7. Comparison Table**

| Feature | Standard Docker | Jetson Containers |
|---------|----------------|-------------------|
| **Runtime** | Default (`runc`) | `--runtime nvidia` |
| **GPU Access** | ❌ None | ✅ CUDA/GPU |
| **NVIDIA Driver Caps** | ❌ None | ✅ `compute,utility,graphics` |
| **Jetson Devices** | ❌ None | ✅ Cameras, I2C, USB, etc. |
| **Jetson Files** | ❌ None | ✅ `/etc/nv_tegra_release`, etc. |
| **Network** | Bridge (default) | `--network host` |

---

## **Summary**

The key differences are:

1. **NVIDIA Container Runtime** - Must be installed and configured as the default runtime
2. **Runtime Flag** - Must use `--runtime nvidia` when running containers
3. **Environment Variables** - Set `NVIDIA_DRIVER_CAPABILITIES` appropriately
4. **Device Mounts** - Mount Jetson-specific hardware devices
5. **Volume Mounts** - Mount Jetson-specific configuration files and sockets

The `jetson-containers` project automates all of this through its `run.sh` script, which automatically detects and mounts all necessary devices and volumes.

---

## **References**

- [jetson-containers Documentation](https://github.com/dusty-nv/jetson-containers)
- [NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-container-runtime)
- [jetson-containers run.sh](https://github.com/dusty-nv/jetson-containers/blob/master/run.sh)
- [jetson-containers setup.md](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md)

