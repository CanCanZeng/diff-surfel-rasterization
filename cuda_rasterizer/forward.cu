/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <iostream>

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ Eigen::Vector3f ComputeColorFromSH(
	int idx,
	int deg,
	int max_coeffs,
	const Eigen::Vector3f& means,
	const Eigen::Vector3f& campos,
	const float* shs,
	bool* clamped
	){
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	Eigen::Vector3f dir = means - campos;
	dir.normalize();

	Eigen::Vector3f* sh = ((Eigen::Vector3f*)shs) + idx * max_coeffs;
	Eigen::Vector3f result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir[0];
		float y = dir[1];
		float z = dir[2];
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += Eigen::Vector3f(0.5f, 0.5f, 0.5f);

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result[0] < 0);
	clamped[3 * idx + 1] = (result[1] < 0);
	clamped[3 * idx + 2] = (result[2] < 0);
	return result.cwiseMax(0.f);
}

__forceinline__ __device__ void GetRect(
	Eigen::Vector2i& rect_min,
	Eigen::Vector2i& rect_max,
	const Eigen::Vector2f& p,
	const int& max_radius,
	const dim3& grid
	){
	rect_min[0] = min(grid.x, max((int)0, (int)((p[0] - max_radius) / BLOCK_X)));
	rect_min[1] = min(grid.y, max((int)0, (int)((p[1] - max_radius) / BLOCK_Y)));

	rect_max[0] = min(grid.x, max((int)0, (int)((p[0] + max_radius + BLOCK_X - 1) / BLOCK_X)));
	rect_max[1] = min(grid.y, max((int)0, (int)((p[1] + max_radius + BLOCK_Y - 1) / BLOCK_Y)));
}


// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void ComputeTransmat(
	const Eigen::Vector3f& p_proj,
	const Eigen::Vector2f& scale,
	const float& mod,
	const Eigen::Vector4f& rot,
	const Eigen::Matrix3f& projmatrix,
	const Eigen::Matrix3f& viewmatrix_R,
	Eigen::Matrix3f& T,
	Eigen::Vector3f& normal
) {
	Eigen::Matrix3f R = Eigen::Quaternionf(rot[0], rot[1], rot[2], rot[3]).matrix();
	Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
	S(0, 0) = scale[0] * mod; S(1, 1) = scale[1] * mod;
	Eigen::Matrix3f L = R * S;

	Eigen::Matrix3f K_Rv_RS = projmatrix * viewmatrix_R * L;
	K_Rv_RS.block<3, 1>(0, 2) = p_proj;  // 前两列保持不变

	T = K_Rv_RS.transpose();  // 公式中用的是 T = (KWH)',即有个转置

	normal = viewmatrix_R * R.col(2);
}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool ComputeAabb(
	const float& sigma2,    // 1^2 or 3^2
	const Eigen::Matrix3f& T,
	Eigen::Vector2f& point_image,
	float& radius
) {
	Eigen::Vector3f T0 = T.col(0);
	Eigen::Vector3f T1 = T.col(1);
	Eigen::Vector3f T2 = T.col(2);
	
	// for x
	Eigen::Vector3f temp_point(sigma2, sigma2, -1.f);
	float a = (T2.cwiseProduct(T2)).dot(temp_point);
	if (abs(a) < 1e-6) {
		return false;
	}

	float a_inv = 1.f / a;

	float b = -2 * (T0.cwiseProduct(T2)).dot(temp_point);
	float c = (T0.cwiseProduct(T0)).dot(temp_point);
	point_image[0] = -b * 0.5 * a_inv;
	float extent_x_square = (b*b - 4*a*c) * a_inv * a_inv * 0.25;
	extent_x_square = extent_x_square > 0 ? extent_x_square : 0;

	// for y
	b = -2 * (T1.cwiseProduct(T2)).dot(temp_point);
	c = (T1.cwiseProduct(T1)).dot(temp_point);
	point_image[1] = -b * 0.5 * a_inv;
	float extent_y_square = (b*b - 4*a*c) * a_inv * a_inv * 0.25;
	extent_y_square = extent_y_square > 0 ? extent_y_square : 0;

	radius = sqrt(extent_x_square + extent_y_square);

	// printf("T[0]=%f %f %f\n", T0[0], T0[1], T0[2]);
	// printf("T[1]=%f %f %f\n", T1[0], T1[1], T1[2]);
	// printf("T[2]=%f %f %f\n", T2[0], T2[1], T2[2]);
	// printf("point_image=%f %f\n", point_image[0], point_image[1]);
	// printf("extent_square=%f %f\n", extent_x_square, extent_y_square);

	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const Eigen::Vector3f* orig_points,
	const Eigen::Vector2f* scales,
	const float scale_modifier,
	const Eigen::Vector4f* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const Eigen::Vector3f* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	Eigen::Vector2f* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	Eigen::Vector4f* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	const Eigen::Matrix4f view_matrix = *(reinterpret_cast<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>*>(viewmatrix));
	// printf("viewmatrix_t=%f %f %f\n", view_matrix(0, 3), view_matrix(1, 3), view_matrix(2, 3));

	Eigen::Matrix3f viewmatrix_R = view_matrix.block<3, 3>(0, 0);
	Eigen::Vector3f viewmatrix_t = view_matrix.block<3, 1>(0, 3);

	const Eigen::Matrix3f proj_matrix = *(reinterpret_cast<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>*>(projmatrix));
	// printf("proj_matrix=\n%f %f %f\n%f %f %f\n%f %f %f\n\n\n", proj_matrix(0, 0), proj_matrix(0, 1), proj_matrix(0, 2), proj_matrix(1, 0), proj_matrix(1, 1), proj_matrix(1, 2), proj_matrix(2, 0), proj_matrix(2, 1), proj_matrix(2, 2));

	Eigen::Matrix<float, 3, 4> a;
	Eigen::Vector3f p_orig = orig_points[idx];
	Eigen::Vector3f p_view = viewmatrix_R * p_orig + viewmatrix_t;
	Eigen::Vector3f p_proj = proj_matrix * p_view;

	// printf("p_orig=%f %f %f\n", p_orig[0], p_orig[1], p_orig[2]);
	// printf("viewmatrix_t=%f %f %f\n", viewmatrix_t[0], viewmatrix_t[1], viewmatrix_t[2]);
	// printf("p_view=%f %f %f\n", p_view[0], p_view[1], p_view[2]);
	// printf("p_proj=%f %f %f\n", p_proj[0]/p_proj[2], p_proj[1]/p_proj[2], p_proj[2]);

	if (!IsInFrustum(p_proj, W, H)) {
		return;
	}

	// Compute transformation matrix
	Eigen::Matrix3f T;
	Eigen::Vector3f normal;
	ComputeTransmat(p_proj, scales[idx], scale_modifier, rotations[idx], proj_matrix, viewmatrix_R, T, normal);
	// printf("T=\n%f %f %f\n%f %f %f\n%f %f %f\n\n\n", T(0, 0), T(0, 1), T(0, 2), T(1, 0), T(1, 1), T(1, 2), T(2, 0), T(2, 1), T(2, 2));
	// printf("normal=%f %f %f\n", normal[0], normal[1], normal[2]);
	for (int row = 0; row<3; row++) {
		for (int col=0; col<3; col++) {
			int j = col * 3 + row;
			transMats[idx * 9 + j] = T(row, col);
		}
	}

	// printf("T=\n%f %f %f\n%f %f %f\n%f %f %f\n",
	// 	transMats[idx*9+0], transMats[idx*9+1], transMats[idx*9+2],
	// 	transMats[idx*9+3], transMats[idx*9+4], transMats[idx*9+5],
	// 	transMats[idx*9+6], transMats[idx*9+7], transMats[idx*9+8]);



	// cull backfacing points
	if (normal.dot(p_view) > -0.01) {
		return;
	}

	// Compute center and radius
	constexpr float sigma2 = 3.f * 3.f; // 原作者采用的是求sigma=1时的半径，然后乘以3。而2dgs-non-official作者采用了直接求sigma=3时的半径，感觉结果更精确，因此采纳后者
	Eigen::Vector2f point_image;
	float radius;
	if (!ComputeAabb(sigma2, T, point_image, radius)) {
		return;
	}

	// printf("point_image=%f %f\n\n\n", point_image[0], point_image[1]);
	
	// Eigen::Vector2i rect_min, rect_max;
	// GetRect(rect_min, rect_max, point_image, radius, grid);
	// if ((rect_max[0] - rect_min[0]) * (rect_max[1] - rect_min[1]) == 0)
	// 	return;

	uint2 rect_min, rect_max;
	getRect(make_float2(point_image[0], point_image[1]), radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors
	Eigen::Vector3f result = ComputeColorFromSH(idx, D, M, p_orig, *cam_pos, shs, clamped);
	rgb[idx * C + 0] = result[0];
	rgb[idx * C + 1] = result[1];
	rgb[idx * C + 2] = result[2];

	depths[idx] = p_view[2];
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = Eigen::Vector4f(normal[0], normal[1], normal[2], opacities[idx]);
	// tiles_touched[idx] = (rect_max[1] - rect_min[1]) * (rect_max[0] - rect_min[0]);
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

	// printf("depths=%f\n", depths[idx]);
	// printf("radius=%d\n", (int)radius);
	// printf("point_image=%f %f\n", point_image[0], point_image[1]);
	// printf("normal_opacity=%f %f %f %f\n", normal[0], normal[1], normal[2], opacities[idx]);
	// printf("tiles_touched=%d\n", tiles_touched[idx]);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (depth < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;
#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		T = fminf(1 - 0.000001, T);

		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others);
}

void FORWARD::preprocess(int P, int D, int M,
	const Eigen::Vector3f* means3D,
	const Eigen::Vector2f* scales,
	const float scale_modifier,
	const Eigen::Vector4f* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const Eigen::Vector3f* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	Eigen::Vector2f* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	Eigen::Vector4f* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	std::cerr << "到了forward.cu里面" << std::endl;
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
	std::cerr << "preprocess处理完了" << std::endl;
}
