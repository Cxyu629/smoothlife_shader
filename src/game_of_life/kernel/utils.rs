pub fn lerp(start: f32, end: f32, t: f32) -> f32 {
    (start * t + end * (1.0 - t)).clamp(0.0, 1.0)
}
