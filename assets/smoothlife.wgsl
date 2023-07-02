@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

struct Params {
    random_float: f32,
    outer_kernel_area: f32,
    inner_kernel_area: f32,
    timestep: f32,
}

@group(0) @binding(1)
var <uniform> params: Params;

@group(0) @binding(2)
var outer_texture: texture_storage_2d<rgba8unorm, read>;

@group(0) @binding(3)
var inner_texture: texture_storage_2d<rgba8unorm, read>;


fn permute4(x: vec4<f32>) -> vec4<f32> { return ((x * 34. + 1.) * x) % vec4<f32>(289.); }
fn fade2(t: vec2<f32>) -> vec2<f32> { return t * t * t * (t * (t * 6. - 15.) + 10.); }
fn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> { return 1.79284291400159 - 0.85373472095314 * r; }
fn fade3(t: vec3<f32>) -> vec3<f32> { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise3(P: vec3<f32>) -> f32 {
  var Pi0 : vec3<f32> = floor(P); // Integer part for indexing
  var Pi1 : vec3<f32> = Pi0 + vec3<f32>(1.); // Integer part + 1
  Pi0 = Pi0 % vec3<f32>(289.);
  Pi1 = Pi1 % vec3<f32>(289.);
  let Pf0 = fract(P); // Fractional part for interpolation
  let Pf1 = Pf0 - vec3<f32>(1.); // Fractional part - 1.
  let ix = vec4<f32>(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  let iy = vec4<f32>(Pi0.yy, Pi1.yy);
  let iz0 = Pi0.zzzz;
  let iz1 = Pi1.zzzz;

  let ixy = permute4(permute4(ix) + iy);
  let ixy0 = permute4(ixy + iz0);
  let ixy1 = permute4(ixy + iz1);

  var gx0: vec4<f32> = ixy0 / 7.;
  var gy0: vec4<f32> = fract(floor(gx0) / 7.) - 0.5;
  gx0 = fract(gx0);
  var gz0: vec4<f32> = vec4<f32>(0.5) - abs(gx0) - abs(gy0);
  var sz0: vec4<f32> = step(gz0, vec4<f32>(0.));
  gx0 = gx0 + sz0 * (step(vec4<f32>(0.), gx0) - 0.5);
  gy0 = gy0 + sz0 * (step(vec4<f32>(0.), gy0) - 0.5);

  var gx1: vec4<f32> = ixy1 / 7.;
  var gy1: vec4<f32> = fract(floor(gx1) / 7.) - 0.5;
  gx1 = fract(gx1);
  var gz1: vec4<f32> = vec4<f32>(0.5) - abs(gx1) - abs(gy1);
  var sz1: vec4<f32> = step(gz1, vec4<f32>(0.));
  gx1 = gx1 - sz1 * (step(vec4<f32>(0.), gx1) - 0.5);
  gy1 = gy1 - sz1 * (step(vec4<f32>(0.), gy1) - 0.5);

  var g000: vec3<f32> = vec3<f32>(gx0.x, gy0.x, gz0.x);
  var g100: vec3<f32> = vec3<f32>(gx0.y, gy0.y, gz0.y);
  var g010: vec3<f32> = vec3<f32>(gx0.z, gy0.z, gz0.z);
  var g110: vec3<f32> = vec3<f32>(gx0.w, gy0.w, gz0.w);
  var g001: vec3<f32> = vec3<f32>(gx1.x, gy1.x, gz1.x);
  var g101: vec3<f32> = vec3<f32>(gx1.y, gy1.y, gz1.y);
  var g011: vec3<f32> = vec3<f32>(gx1.z, gy1.z, gz1.z);
  var g111: vec3<f32> = vec3<f32>(gx1.w, gy1.w, gz1.w);

  let norm0 = taylorInvSqrt4(
      vec4<f32>(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 = g000 * norm0.x;
  g010 = g010 * norm0.y;
  g100 = g100 * norm0.z;
  g110 = g110 * norm0.w;
  let norm1 = taylorInvSqrt4(
      vec4<f32>(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 = g001 * norm1.x;
  g011 = g011 * norm1.y;
  g101 = g101 * norm1.z;
  g111 = g111 * norm1.w;

  let n000 = dot(g000, Pf0);
  let n100 = dot(g100, vec3<f32>(Pf1.x, Pf0.yz));
  let n010 = dot(g010, vec3<f32>(Pf0.x, Pf1.y, Pf0.z));
  let n110 = dot(g110, vec3<f32>(Pf1.xy, Pf0.z));
  let n001 = dot(g001, vec3<f32>(Pf0.xy, Pf1.z));
  let n101 = dot(g101, vec3<f32>(Pf1.x, Pf0.y, Pf1.z));
  let n011 = dot(g011, vec3<f32>(Pf0.x, Pf1.yz));
  let n111 = dot(g111, Pf1);

  var fade_xyz: vec3<f32> = fade3(Pf0);
  let temp = vec4<f32>(f32(fade_xyz.z)); // simplify after chrome bug fix
  let n_z = mix(vec4<f32>(n000, n100, n010, n110), vec4<f32>(n001, n101, n011, n111), temp);
  let n_yz = mix(n_z.xy, n_z.zw, vec2<f32>(f32(fade_xyz.y))); // simplify after chrome bug fix
  let n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let zoom = 30.0;
    
    let color = vec4<f32>(perlinNoise3(vec3<f32>(f32(invocation_id.x)/zoom, f32(invocation_id.y)/zoom, params.random_float)));


    textureStore(texture, location, color);
}

const ra: f32 = 21.0;
const ri: f32 = 7.0;
const b1: f32 = 0.278;
const b2: f32 = 0.365;
const d1: f32 = 0.267;
const d2: f32 = 0.445;
const an: f32 = 0.028;
const am: f32 = 0.147;
const b: f32 = 1.0;
const rate: f32 = 1.0;

fn wrap(coords: vec2<i32>) -> vec2<i32> {
    let dimensions: vec2<i32> = textureDimensions(texture);
    return vec2<i32>(fract(vec2<f32>(coords) / vec2<f32>(dimensions)) * vec2<f32>(dimensions));
}


fn calculate_with_texture(location: vec2<i32>, the_texture: texture_storage_2d<rgba8unorm, read>, the_area: f32, radius: f32) -> vec4<f32> {
    var sum: vec4<f32> = vec4(0.0);
    for (var dx: f32 = -radius; dx <= radius; dx += 1.0) {
        for (var dy: f32 = -radius; dy <= radius; dy += 1.0) {
            let weight = textureLoad(the_texture, wrap(vec2<i32>(i32(radius)) + vec2<i32>(i32(dx), i32(dy))));
            let value = textureLoad(texture, wrap(location + vec2<i32>(i32(dx), i32(dy))));
            sum += value * weight;
        }
    }

    return sum / vec4(the_area);
}

fn logistic_threshold(x: f32, x0: f32, alpha: f32) -> f32 {
    return 1.0 / (1.0 + exp(- 4.0 / alpha * (x - x0)));
}

fn logistic_interval(x: f32, a: f32, b: f32, alpha: f32) -> f32 {
    return logistic_threshold(x, a, alpha) * (1.0 - logistic_threshold(x, b, alpha));
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return (1.0 - t) * a + t * b;
}

fn new_s(n: f32, m: f32) -> f32 {
    let aliveness = logistic_threshold(m, 0.5, am);


    let threshold1 = lerp(b1, d1, aliveness);
    let threshold2 = lerp(b2, d2, aliveness);

    let new_aliveness = logistic_interval(n, threshold1, threshold2, an);

    return clamp(new_aliveness, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let current = textureLoad(texture, location).x;


    let m = calculate_with_texture(location, inner_texture, params.inner_kernel_area, ri+b);
    let n = calculate_with_texture(location, outer_texture, params.outer_kernel_area, ra+b);

    let growth = 2.0 * new_s(n.x,m.x) - 1.0;


    var color = vec4<f32>(current + params.timestep * rate * growth);
    color = clamp(color, vec4(0.0), vec4(1.0));

    storageBarrier();

    textureStore(texture, location, color);
}