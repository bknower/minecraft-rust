// Vertex shader
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0) // 1.
var<uniform> camera: CameraUniform;

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
};


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

//struct SideInput {
//    @location(0) side: u32,
//    @location(1) chunk_coords: vec2<u32>,
//    @location(2) coords: vec3<u32>,
//    @location(3) atlas
//}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var out: VertexOutput;
//    var tex_x = model.atlas_index & 0xff00;
//    var tex_y = model.atlas_index & 0x00ff;
//    out.tex_coords = vec2<f32>(f32(tex_x), f32(tex_y));
    out.tex_coords = model.tex_coords;
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}



// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}


// Create a homogeneous transformation matrix from a translation vector.
//fn mk_translation_matrix(v: vec3<f32>) -> mat4x4<f32>{
//        let c_1: vec4<f32> = vec4<f32>(1., 0., 0., v.x);
//        let c_2: vec4<f32> = vec4<f32>(0., 1., 0., v.y);
//        let c_3: vec4<f32> = vec4<f32>(0., 0., 1., v.z);
//        let c_4: vec4<f32> = vec4<f32>(0., 0., 0., 1.);
//        let translation_matrix = mat4x4<f32>(c_1, c_2, c_3, c_4);
//        return translation_matrix;
//}
//
//fn mk_rotation_matrix(q: vec4<f32>) -> mat4x4<f32> {
//        let m11 = 2. * (q.x * q.x + q.y * q.y) - 1.;
//        let m12 = 2. * (q.y * q.z - q.x * q.w);
//        let m13 = 2. * (q.y * q.w - q.x * q.z);
//
//        let m21 = 2. * (q.y * q.z + q.x * q.w);
//        let m22 = 2. * (q.x * q.x + q.z * q.z) - 1.;
//        let m23 = 2. * (q.z * q.w + q.x * q.y);
//
//        let m31 = 2. * (q.y * q.w - q.x * q.z);
//        let m32 = 2. * (q.z * q.w + q.x * q.y);
//        let m33 = 2. * (q.x * q.x + q.w * q.w) - 1.;
//
//        let c_1: vec4<f32> = vec4<f32>(m11, m21, m31, 0.);
//        let c_2: vec4<f32> = vec4<f32>(m12, m22, m32, 0.);
//        let c_3: vec4<f32> = vec4<f32>(m13, m23, m33, 0.);
//        let c_4: vec4<f32> = vec4<f32>(0., 0., 0., 1.);
//
//
//        let rotation_matrix: mat4x4<f32> = mat4x4<f32>(c_1, c_2, c_3, c_4);
//        return rotation_matrix;
//}
//
//
//fn mat4_mul(A: mat4x4<f32>, B: mat4x4<f32> ) -> mat4x4<f32> {
//
//        // rows of A
//        let r_1: vec4<f32> =  transpose(A)[0];
//        let r_2: vec4<f32> =  transpose(A)[1];
//        let r_3: vec4<f32> =  transpose(A)[2];
//        let r_4: vec4<f32> =  transpose(A)[3];
//        //cols of B
//        let c_1: vec4<f32> = B[0];
//        let c_2: vec4<f32> = B[1];
//        let c_3: vec4<f32> = B[2];
//        let c_4: vec4<f32> = B[3];
//
//        let multiplied = mat4x4<f32>(
//            vec4<f32>(dot(r_1 , c_1), dot(r_2, c_1), dot(r_3, c_1), dot(c_4,c_1)),
//            vec4<f32>(dot(r_1, c_2), dot(r_2, c_2), dot(r_3, c_2), dot(c_4, c_2)),
//            vec4<f32>(dot(r_1, c_3), dot(r_2, c_3), dot(r_3, c_3), dot(c_4, c_3)),
//            vec4<f32>(dot(r_1, c_4), dot(r_2, c_4), dot(r_3, c_4), dot(c_4, c_4)),
//    );
//
//        return multiplied;
//
//}
//
//fn mk_transformation_matrix(position: vec3<f32>, rotation: vec4<f32>) -> mat4x4<f32> {
//    let transformation_matrix = mat4_mul(mk_translation_matrix(position), mk_rotation_matrix(rotation));
//    return transformation_matrix;
//}