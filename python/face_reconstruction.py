import cv2
import numpy as np
from utils import process_uv
from render import MeshRenderer_cpu, MeshRenderer_UV_cpu


def get_colors_from_uv(colors, uv_coords):
    res = bilinear_interpolate(colors, uv_coords[:, 0], uv_coords[:, 1])
    return res

def bilinear_interpolate(img, x, y):
    
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa[..., np.newaxis] * i_a + wb[..., np.newaxis] * i_b + wc[..., np.newaxis] * i_c + wd[..., np.newaxis] * i_d

class face_model:
    def __init__(self, args):
        self.net_recon = cv2.dnn.readNet("weights/net_recon.onnx")
        self.inputsize = 224    ###输入正方形
        model = np.load("weights/face_model.npy",allow_pickle=True).item()
        self.args = args

        # mean shape, size (107127, 1)
        self.u = model["u"].astype(np.float32)
        # face identity bases, size (107127, 80)
        self.id = model["id"].astype(np.float32)
        # face expression bases, size (107127, 64)
        self.exp = model["exp"].astype(np.float32)
        # mean albedo, size (107127, 1)
        self.u_alb = model["u_alb"].astype(np.float32)
        # face albedo bases, size (107127, 80)
        self.alb = model["alb"].astype(np.float32)
        # for computing vertex normals, size (35709, 8)
        self.point_buf = model["point_buf"].astype(np.int64)
        # triangle faces, size (70789, 3)
        self.tri = model["tri"].astype(np.int64)
        # vertex uv coordinates, size (35709, 2), range (0, 1.)
        self.uv_coords = model["uv_coords"].astype(np.float32)

        if args["extractTex"]:
            uv_coords_numpy = process_uv(model["uv_coords"].copy(), 1024, 1024)  # vertex uv coordinates, size (35709, 3)
            self.uv_coords_torch = (uv_coords_numpy.astype(np.float32) / 1023 - 0.5) * 2
            self.uv_renderer = MeshRenderer_UV_cpu(rasterize_size=int(1024.))
            self.uv_coords_torch = self.uv_coords_torch + 1e-6
            self.uv_coords_numpy = uv_coords_numpy.copy()
            self.uv_coords_numpy[:,1] = 1024 - uv_coords_numpy[:,1] - 1

        # vertex indices for 68 landmarks, size (68,)
        if args["ldm68"]:
            self.ldm68 = model["ldm68"].astype(np.int64)
        # vertex indices for 106 landmarks, size (106,)
        if args["ldm106"] or args["ldm106_2d"]:
            self.ldm106 = model["ldm106"].astype(np.int64)
        # vertex indices for 134 landmarks, size (134,)
        if args["ldm134"]:
            self.ldm134 = model["ldm134"].astype(np.int64)

        # segmentation annotation indices for 8 parts, [right_eye, left_eye, right_eyebrow, left_eyebrow, nose, up_lip, down_lip, skin]
        if args["seg_visible"]:
            self.annotation = model["annotation"]  ###里面的每个元素是list, 长度不相同

        # segmentation triangle faces for 8 parts
        if args["seg"]:
            self.annotation_tri = [i.astype(np.int64) for i in model["annotation_tri"]]  ######里面的每个元素是nump.ndarray, 形状不相同

        # face profile parallel, list
        if args["ldm106_2d"]:
            self.parallel = model["parallel"]  ###里面的每个元素是list, 长度不相同
            # parallel for profile matching
            self.v_parallel = - np.ones(35709).astype(np.int64)
            for i in range(len(self.parallel)):
                self.v_parallel[self.parallel[i]]=i

        # focal = 1015, center = 112
        self.persc_proj = np.array([1015.0, 0, 112.0, 0, 1015.0, 112.0, 0, 0, 1], dtype=np.float32).reshape((3, 3)).T
        self.camera_distance = 10.0
        self.init_lit = np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).reshape((1, 1, -1))
        self.SH_a = np.array([np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)], dtype=np.float32)
        self.SH_c = np.array([1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)], dtype=np.float32)

        self.renderer = MeshRenderer_cpu(rasterize_fov=2 * np.arctan(112. / 1015) * 180 / np.pi, znear=5., zfar=15., rasterize_size=int(2 * 112.))

    def split_alpha(self, alpha):
        """
        Return:
            alpha_dict     -- a dict of np.arrays

        Parameters:
            alpha          -- np.array, size (B, 256)
        """
        alpha_id = alpha[:, :80]
        alpha_exp = alpha[:, 80: 144]
        alpha_alb = alpha[:, 144: 224]
        alpha_a = alpha[:, 224: 227]
        alpha_sh = alpha[:, 227: 254]
        alpha_t = alpha[:, 254:]
        return {
            "id": alpha_id,
            "exp": alpha_exp,
            "alb": alpha_alb,
            "angle": alpha_a,
            "sh": alpha_sh,
            "trans": alpha_t
        }
    
    def compute_shape(self, alpha_id, alpha_exp):
        """
        Return:
            face_shape       -- np.array, size (B, N, 3), face vertice without rotation or translation

        Parameters:
            alpha_id         -- np.array, size (B, 80), identity parameter
            alpha_exp        -- np.array, size (B, 64), expression parameter
        """
        batch_size = alpha_id.shape[0]
        face_shape = np.einsum("ij,aj->ai", self.id, alpha_id) + np.einsum("ij,aj->ai", self.exp, alpha_exp) + self.u.reshape((1, -1))
        return face_shape.reshape((batch_size, -1, 3))
    
    def compute_rotation(self, angles):
        """
        Return:
            rot              -- np.array, size (B, 3, 3), pts @ trans_mat

        Parameters:
            angles           -- np.array, size (B, 3), use radian
        """
        batch_size = angles.shape[0]
        ones = np.ones([batch_size, 1])
        zeros = np.zeros([batch_size, 1])
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = np.concatenate([
            ones, zeros, zeros,
            zeros, np.cos(x), -np.sin(x), 
            zeros, np.sin(x), np.cos(x)
        ], axis=1).reshape([batch_size, 3, 3])
        
        rot_y = np.concatenate([
            np.cos(y), zeros, np.sin(y),
            zeros, ones, zeros,
            -np.sin(y), zeros, np.cos(y)
        ], axis=1).reshape([batch_size, 3, 3])

        rot_z = np.concatenate([
            np.cos(z), -np.sin(z), zeros,
            np.sin(z), np.cos(z), zeros,
            zeros, zeros, ones
        ], axis=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        rot = np.transpose(rot, (0, 2, 1))
        return rot
    
    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- np.array, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- np.array, size (B, N, 3)
            rot              -- np.array, size (B, 3, 3)
            trans            -- np.array, size (B, 3)
        """
        return face_shape @ rot + np.expand_dims(trans, axis=1)
    
    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- np.array, size (B, N, 2)

        Parameters:
            face_shape       -- np.array, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]
        return face_proj
    
    def compute_albedo(self, alpha_alb, normalize=True):
        """
        Return:
            face_albedo     -- np.array, size (B, N, 3), in RGB order, range (0, 1.), without lighting

        Parameters:
            alpha_alb        -- np.array, size (B, 80), albedo parameter
        """
        batch_size = alpha_alb.shape[0]
        face_albedo = np.einsum("ij,aj->ai", self.alb, alpha_alb) + self.u_alb.reshape([1, -1])
        if normalize:
            face_albedo = face_albedo / 255.
        return face_albedo.reshape([batch_size, -1, 3])

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- np.array, size (B, N, 3)

        Parameters:
            face_shape       -- np.array, size (B, N, 3)
        """
        v1 = face_shape[:, self.tri[:, 0]]
        v2 = face_shape[:, self.tri[:, 1]]
        v3 = face_shape[:, self.tri[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = np.cross(e1, e2, axis=-1)
        face_norm /= np.linalg.norm(face_norm, ord=2, axis=-1, keepdims=True)
        # face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = np.concatenate([face_norm, np.zeros((face_norm.shape[0], 1, 3))], axis=1)
        
        vertex_norm = np.sum(face_norm[:, self.point_buf], axis=2)
        vertex_norm /= np.linalg.norm(vertex_norm, ord=2, axis=-1, keepdims=True)
        # vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def compute_texture(self, face_albedo, face_norm, alpha_sh):
        """
        Return:
            face_texture        -- np.array, size (B, N, 3), range (0, 1.)

        Parameters:
            face_albedo         -- np.array, size (B, N, 3), from albedo model, range (0, 1.)
            face_norm           -- np.array, size (B, N, 3), rotated face normal
            alpha_sh            -- np.array, size (B, 27), SH parameter
        """
        batch_size = alpha_sh.shape[0]
        v_num = face_albedo.shape[1]
        a = self.SH_a
        c = self.SH_c
        alpha_sh = alpha_sh.reshape([batch_size, 3, 9])
        alpha_sh = alpha_sh + self.init_lit
        alpha_sh = np.transpose(alpha_sh, (0, 2, 1))
        Y = np.concatenate([
             a[0] * c[0] * np.ones_like(face_norm[..., :1]),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], axis=-1)
        r = Y @ alpha_sh[..., :1]
        g = Y @ alpha_sh[..., 1:2]
        b = Y @ alpha_sh[..., 2:]
        face_texture = np.concatenate([r, g, b], axis=-1) * face_albedo
        return face_texture
    
    def add_directionlight(self, normals, lights):
        """
        see https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
        """
        light_direction = lights[:,:,:3]
        light_intensities = lights[:,:,3:]
        directions_to_lights = np.tile(light_direction[:,:,None,:], (1,1,normals.shape[1],1))
        directions_to_lights /= np.linalg.norm(directions_to_lights, ord=2, axis=3, keepdims=True)
        # directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = np.clip((normals[:,None,:,:]*directions_to_lights).sum(axis=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(axis=1)
    
    def compute_gray_shading_with_directionlight(self, face_texture, normals):
        batch_size = normals.shape[0]
        light_positions = np.tile(np.array(
                [
                [-1,1,1],
                [1,1,1],
                [-1,-1,1],
                [1,-1,1],
                [0,0,1]
                ]
        )[None,:,:], (batch_size, 1, 1)).astype(np.float32)
        light_intensities = np.ones_like(light_positions, dtype=np.float32)*1.7
        lights = np.concatenate((light_positions, light_intensities), axis=2)

        shading = self.add_directionlight(normals, lights)
        texture =  face_texture*shading
        return texture
    
    def forward(self, im):
        input_tensor = cv2.dnn.blobFromImage(im.astype(np.float32) / 255.0)
        self.net_recon.setInput(input_tensor)
        # Perform inference on the image
        alpha = self.net_recon.forward(self.net_recon.getUnconnectedOutLayersNames())[0]

        alpha_dict = self.split_alpha(alpha)
        face_shape = self.compute_shape(alpha_dict["id"], alpha_dict["exp"])
        rotation = self.compute_rotation(alpha_dict["angle"])
        face_shape_transformed = self.transform(face_shape, rotation, alpha_dict["trans"])

        # face vertice in 3d
        v3d = self.to_camera(face_shape_transformed)

        # face vertice in 2d image plane
        v2d = self.to_image(v3d)

        # compute face texture with albedo and lighting
        face_albedo = self.compute_albedo(alpha_dict["alb"])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_texture = self.compute_texture(face_albedo, face_norm_roted, alpha_dict["sh"])

        # render shape with texture
        _, _, pred_image, visible_idx_renderer = self.renderer.forward(v3d.copy(), self.tri, np.clip(face_texture, 0, 1).copy(), visible_vertice = True)

        # render shape
        gray_shading = self.compute_gray_shading_with_directionlight(np.ones_like(face_albedo)*0.78, face_norm_roted)
        mask, _, pred_image_shape, _ = self.renderer.forward(v3d.copy(), self.tri, gray_shading.copy())
        
        result_dict = {
            "v3d": v3d,
            "v2d": v2d,
            "face_texture": np.clip(face_texture, 0, 1),
            "tri": self.tri,
            "uv_coords": self.uv_coords,
            "render_shape": np.transpose(pred_image_shape, (0, 2, 3, 1)),
            "render_face": np.transpose(pred_image, (0, 2, 3, 1)),
            "render_mask": np.transpose(mask, (0, 2, 3, 1))}

        # compute visible vertice according to normal and renderer
        if self.args["seg_visible"] or self.args["extractTex"]:
            visible_idx = np.zeros(35709).astype(np.int64)
            visible_idx[visible_idx_renderer.astype(np.int64)] = 1
            visible_idx[(face_norm_roted[..., 2] < 0)[0]] = 0
            # result_dict["visible_idx"] = visible_idx

        # landmarks 68 3d
        if self.args["ldm68"]:
            v2d_68 = self.get_landmarks_68(v2d.copy())
            result_dict["ldm68"] = v2d_68

        # landmarks 106 3d
        if self.args["ldm106"]:
            v2d_106 = self.get_landmarks_106(v2d.copy())
            result_dict["ldm106"] = v2d_106

        # landmarks 106 2d
        if self.args["ldm106_2d"]:
            # v2d_106_2d = self.get_landmarks_106_2d(v2d, face_shape, alpha_dict, visible_idx)
            v2d_106_2d = self.get_landmarks_106_2d(v2d.copy(), face_shape, alpha_dict)
            result_dict["ldm106_2d"] = v2d_106_2d

        # landmarks 134
        if self.args["ldm134"]:
            v2d_134 = self.get_landmarks_134(v2d.copy())
            result_dict["ldm134"] = v2d_134

        # segmentation in 2d without visible mask
        if self.args["seg"]:
            seg = self.segmentation(v3d.copy())
            result_dict["seg"] = seg

        # segmentation in 2d with visible mask
        if self.args["seg_visible"]:
            seg_visible = self.segmentation_visible(v3d.copy(), visible_idx)
            result_dict["seg_visible"] = seg_visible

        # use median-filtered-weight pca-texture for texture blending at invisible region, todo: poisson blending should give better-looking results
        if self.args["extractTex"]:
            _, _, uv_color_pca, _ = self.uv_renderer.forward(self.uv_coords_torch[np.newaxis, ...].copy(), self.tri, (np.clip(face_texture, 0, 1)).copy())
            img_colors = bilinear_interpolate(np.transpose(input_tensor, (0, 2, 3, 1))[0], v2d[0, :, 0], 223 - v2d[0, :, 1])
            _, _, uv_color_img, _ = self.uv_renderer.forward(self.uv_coords_torch[np.newaxis, ...].copy(), self.tri, img_colors[np.newaxis, ...].copy())
            _, _, uv_weight, _ = self.uv_renderer.forward(self.uv_coords_torch[np.newaxis, ...].copy(), self.tri, (1 - np.stack((visible_idx,)*3, axis=-1)[np.newaxis, ...].astype(np.float32)).copy())

            median_filtered_w = cv2.medianBlur((np.transpose(uv_weight, (0, 2, 3, 1))[0]*255).astype(np.uint8), 31)/255.

            uv_color_pca = np.transpose(uv_color_pca, (0, 2, 3, 1))[0]
            uv_color_img = np.transpose(uv_color_img, (0, 2, 3, 1))[0]

            res_colors = ((1 - median_filtered_w) * np.clip(uv_color_img, 0, 1) + median_filtered_w * np.clip(uv_color_pca, 0, 1))
            # result_dict["extractTex_uv"] = res_colors
            v_colors = get_colors_from_uv(res_colors.copy(), self.uv_coords_numpy.copy())
            result_dict["extractTex"] = v_colors

        return result_dict


    def get_landmarks_68(self, v2d):
        """
        Return:
            landmarks_68_3d         -- np.array, size (B, 68, 2)

        Parameters:
            v2d                     -- np.array, size (B, N, 2)
        """
        return v2d[:, self.ldm68]

    def get_landmarks_106(self, v2d):
        """
        Return:
            landmarks_106_3d         -- np.array, size (B, 106, 2)

        Parameters:
            v2d                      -- np.array, size (B, N, 2)
        """
        return v2d[:, self.ldm106]
    
    def get_landmarks_134(self, v2d):
        """
        Return:
            landmarks_134            -- np.array, size (B, 134, 2)

        Parameters:
            v2d                      -- np.array, size (B, N, 2)
        """
        return v2d[:, self.ldm134]

    def get_landmarks_106_2d(self, v2d, face_shape, alpha_dict):
        """
        Return:
            landmarks_106_2d         -- np.array, size (B, 106, 2)

        Parameters:
            v2d                     -- np.array, size (B, N, 2)
            face_shape              -- np.array, size (B, N, 3), face vertice without rotation or translation
            alpha_dict              -- a dict of np.arrays
        """

        temp_angle = alpha_dict["angle"].copy()
        temp_angle[:,2] = 0
        rotation_without_roll = self.compute_rotation(temp_angle)
        v2d_without_roll = self.to_image(self.to_camera(self.transform(face_shape, rotation_without_roll, alpha_dict["trans"])))

        visible_parallel = self.v_parallel.copy()
        # visible_parallel[visible_idx == 0] = -1

        ldm106_dynamic=self.ldm106.copy()
        for i in range(16):
            temp=v2d_without_roll.copy()[:,:,0]
            temp[:,visible_parallel!=i] = 1e5
            ldm106_dynamic[i]=np.argmin(temp)

        for i in range(17,33):
            temp=v2d_without_roll.copy()[:,:,0]
            temp[:,visible_parallel!=i] = -1e5
            ldm106_dynamic[i]=np.argmax(temp)

        return v2d[:, ldm106_dynamic]
    
    def segmentation(self, v3d):

        seg = np.zeros((224,224,8))
        for i in range(8):
            mask, _, _, _ = self.renderer.forward(v3d.copy(), self.annotation_tri[i])
            seg[:,:,i] = mask.squeeze()
        return seg

    def segmentation_visible(self, v3d, visible_idx):

        seg = np.zeros((224,224,8))
        for i in range(8):
            temp = np.zeros_like(v3d)
            temp[:,self.annotation[i],:] = 1
            temp[:,visible_idx == 0,:] = 0
            _, _, temp_image, _ = self.renderer.forward(v3d.copy(), self.tri, temp.copy())
            temp_image = temp_image.mean(axis=1)
            mask = np.where(temp_image >= 0.5, 1.0, 0.0)
            seg[:,:,i] = mask.squeeze()
        return seg
