# PoseCartoonGenerator

A general-purpose pipeline for generating webcomic-style images from stick figure pose definitions.

## Architecture

1. Define a minimal stick figure skeleton (graph of joints + bones)
2. Specify poses as joint angles (forward kinematics)
3. Render skeletons to conditioning images
4. Use ControlNet to generate styled comic characters from skeletons
5. Composite characters into panel layouts with text/dialogue

The core idea: separate POSE (what the character is doing) from STYLE (how it looks). Pose is a tiny parameter space (~10 angles). Style is handled by the diffusion model + LoRA fine-tune.

## Requirements

```bash
pip install diffusers transformers accelerate torch torchvision
pip install pillow numpy
```

## Usage

### Render a Single Pose

```bash
python pose_detector.py render --torso 0 --l-shoulder 20 --r-shoulder -20 --out pose.png
```

### Generate Training Data

```bash
python pose_detector.py generate-data --output-dir ./training_data --num-samples 1000
```

## Classes

- `JointType`: Enum for joint types
- `Joint`: Dataclass for joint positions
- `BodyProportions`: Dataclass for body measurements
- `Pose`: Dataclass for pose angles
- `SkeletonSolver`: Converts poses to joint positions
- `SkeletonRenderer`: Renders skeletons to images
- `ComicGenerator`: Uses ControlNet to generate styled characters
- `Compositor`: Assembles panels with characters and text

## License

MIT License