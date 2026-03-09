"""
skeleton_comic_gen.py

General-purpose pipeline for generating webcomic-style images from
stick figure pose definitions.

Architecture:
    1. Define a minimal stick figure skeleton (graph of joints + bones)
    2. Specify poses as joint angles (forward kinematics)
    3. Render skeletons to conditioning images
    4. Use ControlNet to generate styled comic characters from skeletons
    5. Composite characters into panel layouts with text/dialogue

The core idea: separate POSE (what the character is doing) from STYLE
(how it looks). Pose is a tiny parameter space (~10 angles). Style is
handled by the diffusion model + LoRA fine-tune.

Requirements:
    pip install diffusers transformers accelerate torch torchvision
    pip install pillow numpy
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont



class JointType(Enum):
    HEAD        = "head"
    NECK        = "neck"
    L_SHOULDER  = "l_shoulder"
    R_SHOULDER  = "r_shoulder"
    L_ELBOW     = "l_elbow"
    R_ELBOW     = "r_elbow"
    L_HAND      = "l_hand"
    R_HAND      = "r_hand"
    HIP_CENTER  = "hip_center"
    L_HIP       = "l_hip"
    R_HIP       = "r_hip"
    L_KNEE      = "l_knee"
    R_KNEE      = "r_knee"
    L_FOOT      = "l_foot"
    R_FOOT      = "r_foot"


BONES = [
    (JointType.NECK, JointType.HIP_CENTER),
    (JointType.NECK, JointType.L_SHOULDER),
    (JointType.NECK, JointType.R_SHOULDER),
    (JointType.L_SHOULDER, JointType.L_ELBOW),
    (JointType.L_ELBOW, JointType.L_HAND),
    (JointType.R_SHOULDER, JointType.R_ELBOW),
    (JointType.R_ELBOW, JointType.R_HAND),
    (JointType.HIP_CENTER, JointType.L_HIP),
    (JointType.HIP_CENTER, JointType.R_HIP),
    (JointType.L_HIP, JointType.L_KNEE),
    (JointType.L_KNEE, JointType.L_FOOT),
    (JointType.R_HIP, JointType.R_KNEE),
    (JointType.R_KNEE, JointType.R_FOOT),
]


@dataclass
class Joint:
    joint_type: JointType
    x: float  
    y: float  
    visible: bool = True


@dataclass
class BodyProportions:
    """
    All lengths relative to total figure height (0.0 - 1.0).
    Adjust these to create different body types (stocky, lanky, chibi, etc.)
    """
    head_radius: float = 0.08
    neck_length: float = 0.05
    torso_length: float = 0.25
    upper_arm_length: float = 0.15
    forearm_length: float = 0.13
    upper_leg_length: float = 0.20
    lower_leg_length: float = 0.18
    shoulder_width: float = 0.15   
    hip_width: float = 0.08       


@dataclass
class Pose:
    """
    A pose is fully defined by joint angles (degrees).
    Convention: 0° = straight down, positive = clockwise.

    This is the entire "language" for character posing — every
    possible character pose maps to a point in this ~10D space.
    """
    torso_angle: float = 0.0

    l_shoulder_angle: float = 20.0
    r_shoulder_angle: float = -20.0
    l_elbow_angle: float = 0.0
    r_elbow_angle: float = 0.0

    l_hip_angle: float = 5.0
    r_hip_angle: float = -5.0
    l_knee_angle: float = 0.0
    r_knee_angle: float = 0.0

    head_tilt: float = 0.0

    def perturb(self, sigma: float = 10.0) -> "Pose":
        """Return a slightly varied copy (useful for generating training data)."""
        return Pose(**{
            f.name: getattr(self, f.name) + random.gauss(0, sigma)
            for f in self.__dataclass_fields__.values()
        })



class SkeletonSolver:
    """Convert a Pose (angles) + BodyProportions into absolute joint positions."""

    def __init__(self, proportions: Optional[BodyProportions] = None):
        self.proportions = proportions or BodyProportions()

    def solve(
        self,
        pose: Pose,
        origin_x: float = 0.5,
        origin_y: float = 0.15,
        scale: float = 1.0,
    ) -> dict[JointType, Joint]:
        p = self.proportions
        joints = {}

        def polar(angle_deg, length):
            rad = math.radians(angle_deg)
            return math.sin(rad) * length * scale, math.cos(rad) * length * scale

        
        joints[JointType.NECK] = Joint(JointType.NECK, origin_x, origin_y)

        
        hx = origin_x + math.sin(math.radians(pose.head_tilt)) * p.neck_length * scale
        hy = origin_y - p.neck_length * scale
        joints[JointType.HEAD] = Joint(JointType.HEAD, hx, hy)

        
        tdx, tdy = polar(pose.torso_angle, p.torso_length)
        hip_x, hip_y = origin_x + tdx, origin_y + tdy
        joints[JointType.HIP_CENTER] = Joint(JointType.HIP_CENTER, hip_x, hip_y)

        
        perp = pose.torso_angle + 90
        pdx, pdy = polar(perp, p.shoulder_width)
        joints[JointType.L_SHOULDER] = Joint(JointType.L_SHOULDER, origin_x + pdx, origin_y + pdy)
        joints[JointType.R_SHOULDER] = Joint(JointType.R_SHOULDER, origin_x - pdx, origin_y - pdy)

        
        for side, sh_jt, el_jt, ha_jt, sh_angle, el_angle in [
            ("L", JointType.L_SHOULDER, JointType.L_ELBOW, JointType.L_HAND,
             pose.l_shoulder_angle, pose.l_elbow_angle),
            ("R", JointType.R_SHOULDER, JointType.R_ELBOW, JointType.R_HAND,
             pose.r_shoulder_angle, pose.r_elbow_angle),
        ]:
            sh = joints[sh_jt]
            ua_angle = pose.torso_angle + sh_angle
            ua_dx, ua_dy = polar(ua_angle, p.upper_arm_length)
            ex, ey = sh.x + ua_dx, sh.y + ua_dy
            joints[el_jt] = Joint(el_jt, ex, ey)

            fa_angle = ua_angle + el_angle
            fa_dx, fa_dy = polar(fa_angle, p.forearm_length)
            joints[ha_jt] = Joint(ha_jt, ex + fa_dx, ey + fa_dy)

        
        hpdx, hpdy = polar(perp, p.hip_width)
        joints[JointType.L_HIP] = Joint(JointType.L_HIP, hip_x + hpdx, hip_y + hpdy)
        joints[JointType.R_HIP] = Joint(JointType.R_HIP, hip_x - hpdx, hip_y - hpdy)

        
        for side, hip_jt, kn_jt, ft_jt, hip_angle, kn_angle in [
            ("L", JointType.L_HIP, JointType.L_KNEE, JointType.L_FOOT,
             pose.l_hip_angle, pose.l_knee_angle),
            ("R", JointType.R_HIP, JointType.R_KNEE, JointType.R_FOOT,
             pose.r_hip_angle, pose.r_knee_angle),
        ]:
            h = joints[hip_jt]
            ul_angle = pose.torso_angle + hip_angle
            ul_dx, ul_dy = polar(ul_angle, p.upper_leg_length)
            kx, ky = h.x + ul_dx, h.y + ul_dy
            joints[kn_jt] = Joint(kn_jt, kx, ky)

            ll_angle = ul_angle + kn_angle
            ll_dx, ll_dy = polar(ll_angle, p.lower_leg_length)
            joints[ft_jt] = Joint(ft_jt, kx + ll_dx, ky + ll_dy)

        return joints



class SkeletonRenderer:
    """
    Render joint positions to an image for ControlNet conditioning.

    Modes:
        - "openpose":     Colored bones on black (for OpenPose ControlNet)
        - "stickfigure":  Black on white (for Canny / lineart ControlNet)
    """

    BONE_GROUPS = {
        frozenset({"neck", "hip_center"}): (255, 0, 0),
        frozenset({"neck", "l_shoulder"}): (0, 255, 0),
        frozenset({"neck", "r_shoulder"}): (0, 0, 255),
    }

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height

    def _px(self, joint: Joint) -> tuple[int, int]:
        return int(joint.x * self.width), int(joint.y * self.height)

    def render(
        self,
        joints: dict[JointType, Joint],
        mode: str = "stickfigure",
        line_width: Optional[int] = None,
    ) -> Image.Image:
        if mode == "openpose":
            return self._render_openpose(joints, line_width)
        return self._render_stickfigure(joints, line_width)

    def _render_stickfigure(self, joints, line_width) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        lw = line_width or max(3, self.width // 80)

        for jt_a, jt_b in BONES:
            if jt_a in joints and jt_b in joints:
                a, b = joints[jt_a], joints[jt_b]
                if a.visible and b.visible:
                    draw.line([self._px(a), self._px(b)], fill=(0, 0, 0), width=lw)

        if JointType.HEAD in joints:
            hd = joints[JointType.HEAD]
            r = max(10, self.width // 12)
            px, py = self._px(hd)
            draw.ellipse([px - r, py - r, px + r, py + r], outline=(0, 0, 0), width=lw)

        return img

    def _render_openpose(self, joints, line_width) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        lw = line_width or max(2, self.width // 100)

        
        def color_for(jt_a, jt_b):
            names = jt_a.value + jt_b.value
            if "l_" in names and ("shoulder" in names or "elbow" in names or "hand" in names):
                return (0, 255, 0)
            if "r_" in names and ("shoulder" in names or "elbow" in names or "hand" in names):
                return (0, 0, 255)
            if "l_" in names:
                return (255, 255, 0)
            if "r_" in names:
                return (255, 0, 255)
            return (255, 0, 0)

        for jt_a, jt_b in BONES:
            if jt_a in joints and jt_b in joints:
                a, b = joints[jt_a], joints[jt_b]
                if a.visible and b.visible:
                    draw.line([self._px(a), self._px(b)], fill=color_for(jt_a, jt_b), width=lw)

        if JointType.HEAD in joints:
            hd = joints[JointType.HEAD]
            r = max(8, self.width // 15)
            px, py = self._px(hd)
            draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 255, 255), width=lw)

        dot_r = max(2, self.width // 150)
        for jt, j in joints.items():
            if j.visible and jt != JointType.HEAD:
                px, py = self._px(j)
                draw.ellipse([px - dot_r, py - dot_r, px + dot_r, py + dot_r], fill=(255, 255, 255))

        return img



class ComicGenerator:
    """
    Skeleton image → ControlNet → styled comic character.

    The skeleton handles structure. The prompt + LoRA handle style.
    """

    DEFAULT_STYLE = (
        "black and white webcomic, thick outlines, simple cartoon character, "
        "minimal shading, clean lines, white background"
    )
    DEFAULT_NEGATIVE = (
        "color, gradient, shading, realistic, photograph, blurry, "
        "low quality, watermark"
    )

    def __init__(
        self,
        controlnet_model: str = "lllyasviel/control_v11p_sd15_openpose",
        base_model: str = "runwayml/stable-diffusion-v1-5",
        lora_path: Optional[str] = None,
        device: str = "cuda",
    ):
        import torch
        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        if lora_path:
            self.pipe.unet.load_attn_procs(lora_path)

        self.solver = SkeletonSolver()
        self.renderer = SkeletonRenderer()
        self.device = device

    def generate(
        self,
        pose: Pose,
        prompt: str = "",
        negative_prompt: str = "",
        proportions: Optional[BodyProportions] = None,
        origin: tuple[float, float] = (0.5, 0.15),
        scale: float = 1.0,
        render_mode: str = "openpose",
        controlnet_scale: float = 1.0,
        guidance_scale: float = 7.5,
        steps: int = 30,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Generate a comic character from a pose.
        Returns (generated_image, conditioning_skeleton).
        """
        import torch

        if proportions:
            self.solver.proportions = proportions

        joints = self.solver.solve(pose, origin_x=origin[0], origin_y=origin[1], scale=scale)

        self.renderer.width = width
        self.renderer.height = height
        cond_img = self.renderer.render(joints, mode=render_mode)

        full_prompt = f"{prompt}, {self.DEFAULT_STYLE}" if prompt else self.DEFAULT_STYLE
        neg = negative_prompt or self.DEFAULT_NEGATIVE

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=neg,
            image=cond_img,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
            width=width,
            height=height,
        )

        return result.images[0], cond_img



@dataclass
class Character:
    """A character placed in a panel."""
    pose: Pose
    x: float = 0.5          
    y: float = 0.25
    scale: float = 0.7
    proportions: Optional[BodyProportions] = None


@dataclass
class SpeechBubble:
    text: str
    x: float                 
    y: float


@dataclass
class Panel:
    characters: list[Character] = field(default_factory=list)
    bubbles: list[SpeechBubble] = field(default_factory=list)
    caption: str = ""


class Compositor:
    """
    Assemble characters + text into comic panel layouts.

    Handles:
        - Multi-character scenes
        - Speech bubbles with word wrap
        - Panel borders and gutters
        - Horizontal strip or grid layouts
    """

    def __init__(
        self,
        panel_w: int = 400,
        panel_h: int = 350,
        gutter: int = 10,
        border: int = 3,
    ):
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.gutter = gutter
        self.border = border
        self.solver = SkeletonSolver()

    def compose(
        self,
        panels: list[Panel],
        title: str = "",
        columns: Optional[int] = None,
    ) -> Image.Image:
        """
        Render panels into a comic strip or grid.
        If columns is None, renders as a single horizontal row.
        """
        n = len(panels)
        cols = columns or n
        rows = math.ceil(n / cols)

        title_h = 50 if title else 0
        total_w = cols * self.panel_w + (cols + 1) * self.gutter
        total_h = rows * self.panel_h + (rows + 1) * self.gutter + title_h

        img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        if title:
            font = self._font(20, bold=True)
            bbox = draw.textbbox((0, 0), title, font=font)
            tx = (total_w - (bbox[2] - bbox[0])) // 2
            draw.text((tx, 12), title, fill=(0, 0, 0), font=font)

        for i, panel in enumerate(panels):
            col, row = i % cols, i // cols
            px = self.gutter + col * (self.panel_w + self.gutter)
            py = self.gutter + title_h + row * (self.panel_h + self.gutter)

            
            draw.rectangle(
                [px, py, px + self.panel_w, py + self.panel_h],
                outline=(0, 0, 0), width=self.border,
            )

            
            if panel.caption:
                font = self._font(12)
                draw.text((px + 10, py + 8), panel.caption, fill=(0, 0, 0), font=font)

            
            renderer = SkeletonRenderer(self.panel_w, self.panel_h)
            for char in panel.characters:
                solver = SkeletonSolver(char.proportions or BodyProportions())
                joints = solver.solve(char.pose, origin_x=char.x, origin_y=char.y, scale=char.scale)
                char_img = renderer.render(joints, mode="stickfigure", line_width=3)

                
                ca = np.array(char_img)
                sa = np.array(img)
                mask = ca[:, :, 0] < 128
                for cy in range(self.panel_h):
                    for cx in range(self.panel_w):
                        if mask[cy, cx]:
                            sx, sy = px + cx, py + cy
                            if 0 <= sx < total_w and 0 <= sy < total_h:
                                sa[sy, sx] = ca[cy, cx]
                img = Image.fromarray(sa)
                draw = ImageDraw.Draw(img)

            
            for bubble in panel.bubbles:
                bx = px + int(bubble.x * self.panel_w)
                by = py + int(bubble.y * self.panel_h)
                self._draw_bubble(draw, bubble.text, bx, by)

        return img

    def _draw_bubble(self, draw, text, x, y, padding=8, max_w=150):
        font = self._font(11)
        lines = self._wrap(draw, text, font, max_w)
        lh = 15
        tw = max(
            (draw.textbbox((0, 0), ln, font=font)[2] - draw.textbbox((0, 0), ln, font=font)[0])
            for ln in lines
        )
        th = len(lines) * lh

        bx, by_ = x - tw // 2 - padding, y - th - padding * 2
        bw, bh = tw + padding * 2, th + padding * 2

        draw.rounded_rectangle(
            [bx, by_, bx + bw, by_ + bh], radius=8,
            outline=(0, 0, 0), fill=(255, 255, 255), width=2,
        )
        draw.polygon(
            [(x - 5, by_ + bh), (x + 5, by_ + bh), (x, by_ + bh + 10)],
            fill=(255, 255, 255), outline=(0, 0, 0),
        )
        for j, ln in enumerate(lines):
            draw.text((bx + padding, by_ + padding + j * lh), ln, fill=(0, 0, 0), font=font)

    def _wrap(self, draw, text, font, max_w):
        words, lines, cur = text.split(), [], ""
        for w in words:
            test = f"{cur} {w}".strip()
            if (draw.textbbox((0, 0), test, font=font)[2] - draw.textbbox((0, 0), test, font=font)[0]) > max_w:
                if cur:
                    lines.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            lines.append(cur)
        return lines or [""]

    def _font(self, size, bold=False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
        except OSError:
            return ImageFont.load_default()



def generate_training_pairs(
    output_dir: str,
    num_samples: int = 1000,
    resolution: int = 512,
):
    """
    Generate (skeleton, prompt) pairs for ControlNet fine-tuning.

    Randomly samples from the full pose space with varying body
    proportions to cover diverse character types and actions.
    """
    import json, os

    os.makedirs(f"{output_dir}/skeletons", exist_ok=True)
    os.makedirs(f"{output_dir}/openpose", exist_ok=True)

    solver = SkeletonSolver()
    renderer = SkeletonRenderer(resolution, resolution)
    metadata = []

    for i in range(num_samples):
        
        pose = Pose(
            torso_angle=random.gauss(0, 15),
            l_shoulder_angle=random.uniform(-180, 180),
            r_shoulder_angle=random.uniform(-180, 180),
            l_elbow_angle=random.uniform(-150, 150),
            r_elbow_angle=random.uniform(-150, 150),
            l_hip_angle=random.gauss(0, 25),
            r_hip_angle=random.gauss(0, 25),
            l_knee_angle=random.uniform(-120, 10),
            r_knee_angle=random.uniform(-120, 10),
            head_tilt=random.gauss(0, 15),
        )

        
        proportions = BodyProportions(
            head_radius=random.uniform(0.05, 0.12),
            torso_length=random.uniform(0.18, 0.30),
            upper_arm_length=random.uniform(0.10, 0.18),
            forearm_length=random.uniform(0.08, 0.16),
            upper_leg_length=random.uniform(0.15, 0.25),
            lower_leg_length=random.uniform(0.13, 0.22),
            shoulder_width=random.uniform(0.10, 0.20),
            hip_width=random.uniform(0.05, 0.12),
        )

        solver.proportions = proportions
        joints = solver.solve(pose)

        renderer.width = resolution
        renderer.height = resolution

        stick = renderer.render(joints, mode="stickfigure")
        openpose = renderer.render(joints, mode="openpose")

        name = f"{i:06d}"
        stick.save(f"{output_dir}/skeletons/{name}.png")
        openpose.save(f"{output_dir}/openpose/{name}.png")

        metadata.append({
            "file": f"{name}.png",
            "pose": {k: round(v, 2) for k, v in pose.__dict__.items()},
            "proportions": {k: round(v, 4) for k, v in proportions.__dict__.items()},
        })

    with open(f"{output_dir}/metadata.jsonl", "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {num_samples} training pairs in {output_dir}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    
    render_p = sub.add_parser("render", help="Render a pose to image")
    render_p.add_argument("--torso", type=float, default=0)
    render_p.add_argument("--l-shoulder", type=float, default=20)
    render_p.add_argument("--r-shoulder", type=float, default=-20)
    render_p.add_argument("--l-elbow", type=float, default=0)
    render_p.add_argument("--r-elbow", type=float, default=0)
    render_p.add_argument("--l-hip", type=float, default=5)
    render_p.add_argument("--r-hip", type=float, default=-5)
    render_p.add_argument("--l-knee", type=float, default=0)
    render_p.add_argument("--r-knee", type=float, default=0)
    render_p.add_argument("--head", type=float, default=0)
    render_p.add_argument("--mode", choices=["stickfigure", "openpose"], default="stickfigure")
    render_p.add_argument("--size", type=int, default=512)
    render_p.add_argument("--out", type=str, default="pose.png")

    
    data_p = sub.add_parser("generate-data", help="Generate training pairs")
    data_p.add_argument("--output-dir", type=str, default="./training_data")
    data_p.add_argument("--num-samples", type=int, default=1000)
    data_p.add_argument("--resolution", type=int, default=512)

    args = parser.parse_args()

    if args.command == "render":
        pose = Pose(
            torso_angle=args.torso,
            l_shoulder_angle=args.l_shoulder,
            r_shoulder_angle=args.r_shoulder,
            l_elbow_angle=args.l_elbow,
            r_elbow_angle=args.r_elbow,
            l_hip_angle=args.l_hip,
            r_hip_angle=args.r_hip,
            l_knee_angle=args.l_knee,
            r_knee_angle=args.r_knee,
            head_tilt=args.head,
        )
        solver = SkeletonSolver()
        renderer = SkeletonRenderer(args.size, args.size)
        joints = solver.solve(pose)
        img = renderer.render(joints, mode=args.mode)
        img.save(args.out)
        print(f"Saved {args.out}")

    elif args.command == "generate-data":
        generate_training_pairs(args.output_dir, args.num_samples, args.resolution)