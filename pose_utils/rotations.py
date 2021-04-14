import torch
import numpy as np
import numpy.testing as npt


def normalize(q):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1, keepdim=True)
    return q.div(norm)


def assert_normalized(q, atol=1e-3):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1)
    norm_check =  (norm - 1.0).abs()
    try:
        assert torch.max(norm_check) < atol
    except:
        print("normalization failure: {}.".format(torch.max(norm_check)))
        return -1
    return 0


def multiply(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    assert q.shape == r.shape

    real1, im1 = q.split([1, 3], dim=-1)
    real2, im2 = r.split([1, 3], dim=-1)

    real = real1*real2 -  torch.sum(im1*im2, dim=-1, keepdim=True) 
    im = real1*im2 + real2*im1 + im1.cross(im2)
    return torch.cat((real, im), dim=-1)


def conjugate(q):
    assert q.shape[-1] == 4
    w, xyz = q.split([1, 3], dim=-1)
    return torch.cat((w, -xyz), dim=-1)


def rotate(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    assert_normalized(q)
    
    zeros = torch.zeros(original_shape[:-1]+[1])
    qv = torch.cat((zeros, v), dim=-1)

    result = multiply(multiply(q, qv), conjugate(q))
    _, xyz = result.split([1, 3], dim=-1)
    return xyz.view(original_shape)


def relative_angle(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    assert q.shape == r.shape

    assert_normalized(q)
    assert_normalized(r)

    dot_product = torch.sum(q*r, dim=-1)
    angle = 2 * torch.acos(torch.clamp(dot_product.abs(), min=-1, max=1))

    return angle


def unit_quaternion_to_matrix(q):
    assert_normalized(q)
    w, x, y, z= torch.unbind(q, dim=-1)
    matrix = torch.stack(( 1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y* w,
                        2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w,  
                        2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x -2*y*y),
                        dim=-1)
    matrix_shape = list(matrix.shape)[:-1]+[3,3]
    return matrix.view(matrix_shape).contiguous()


def matrix_to_unit_quaternion(matrix):
    assert matrix.shape[-1] == matrix.shape[-2] == 3
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)

    trace = 1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    trace = torch.clamp(trace, min=0.)
    r = torch.sqrt(trace)
    s = 1.0 / (2 * r + 1e-7)
    w = 0.5 * r
    x = (matrix[..., 2, 1] - matrix[..., 1, 2])*s
    y = (matrix[..., 0, 2] - matrix[..., 2, 0])*s
    z = (matrix[..., 1, 0] - matrix[..., 0, 1])*s

    q = torch.stack((w, x, y, z), dim=-1)

    return normalize(q)


def axis_theta_to_quater(axis, theta):  # axis: [Bs, 3], theta: [Bs]
    w = torch.cos(theta / 2.)  # [Bs]
    u = torch.sin(theta / 2.)  # [Bs]
    xyz = axis * u.unsqueeze(-1)  # [Bs, 3]
    new_q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)  # [Bs, 4]
    new_q = normalize(new_q)
    return new_q


def quater_to_axis_theta(quater):
    quater = normalize(quater)
    cosa = quater[..., 0]
    sina = torch.sqrt(1 - cosa ** 2)
    norm = sina.unsqueeze(-1)
    mask = (norm < 1e-8).float()
    axis = quater[..., 1:] / torch.max(norm, mask)
    theta = 2 * torch.acos(torch.clamp(cosa, min=-1, max=1))
    return axis, theta


def axis_theta_to_matrix(axis, theta):
    quater = axis_theta_to_quater(axis, theta)  # [Bs, 4]
    return unit_quaternion_to_matrix(quater)


def matrix_to_axis_theta(matrix):
    quater = matrix_to_unit_quaternion(matrix)
    return quater_to_axis_theta(quater)


def matrix_to_rotvec(matrix):
    axis, theta = matrix_to_axis_theta(matrix)
    theta = theta % (2 * np.pi) + 2 * np.pi
    return axis * theta.unsqueeze(-1)


def rotvec_to_axis_theta(rotvec):
    theta = torch.norm(rotvec, dim=-1, keepdim=True)  # [Bs, 1]
    mask = (theta < 1e-8).float()
    axis = rotvec / torch.max(theta, mask)  # [Bs, 3]
    theta = theta.squeeze(-1)  # [Bs]
    return axis, theta


def rotvec_to_matrix(rotvec):  # [Bs, 3]
    axis, theta = rotvec_to_axis_theta(rotvec)
    return axis_theta_to_matrix(axis, theta)


def rotvec_to_euler(rotvec):
    """
    http://euclideanspace.com/maths/geometry/rotations/conversions/angleToEuler/index.htm
    """
    axis, theta = rotvec_to_axis_theta(rotvec)
    x, y, z = torch.unbind(axis, dim=-1)
    s, c = torch.sin(theta), torch.cos(theta)
    t = 1 - c

    mask_n = ((x * y * t + z * s) > 0.998).float().unsqueeze(-1)
    heading = 2 * torch.atan2(x * torch.sin(theta / 2),
                              torch.cos(theta / 2))
    attitude = torch.ones_like(heading) * np.pi / 2.0
    bank = torch.zeros_like(heading)
    euler_n = torch.stack([heading, attitude, bank], dim=-1)

    mask_s = ((x * y * t + z * s) < -0.998).float().unsqueeze(-1)
    heading = -2 * torch.atan2(x * torch.sin(theta / 2),
                              torch.cos(theta / 2))
    attitude = -torch.ones_like(heading) * np.pi / 2.0
    bank = torch.zeros_like(heading)
    euler_s = torch.stack([heading, attitude, bank], dim=-1)

    heading = torch.atan2(y * s - x * z * t, 1 - (y * y + z * z) * t)
    attitude = torch.asin(x * y * t + z * s)
    bank = torch.atan2(x * s - y * z * t, 1 - (x * x + z * z) * t)
    euler = torch.stack([heading, attitude, bank], dim=-1)
    mask = torch.ones_like(mask_n) - mask_n - mask_s

    euler_final = mask_n * euler_n + mask_s * euler_s + mask * euler

    return euler_final


def euler_to_rotvec(euler):
    """
    http://euclideanspace.com/maths/geometry/rotations/conversions/eulerToAngle/index.htm
    """
    heading, attitude, bank = torch.unbind(euler, dim=-1)
    c1 = torch.cos(heading / 2)
    s1 = torch.sin(heading / 2)
    c2 = torch.cos(attitude / 2)
    s2 = torch.sin(attitude / 2)
    c3 = torch.cos(bank / 2)
    s3 = torch.sin(bank / 2)
    c1c2 = c1 * c2
    s1s2 = s1 * s2
    w = c1c2 * c3 - s1s2 * s3
    x = c1c2 * s3 + s1s2 * c3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    angle = 2 * torch.acos(w)
    axis = torch.stack([x, y, z], dim=-1)
    norm = torch.norm(axis, dim=-1, keepdim=True)
    mask = (norm < 1e-8).float()
    axis = axis / torch.max(norm, mask)
    u_axis = torch.zeros_like(axis)
    u_axis[..., 0] = 1.0
    axis_final = mask * u_axis + (1 - mask) * axis
    return axis_final * angle.unsqueeze(-1)


def jitter_quaternion(q, theta):  #[Bs, 4], [Bs, 1]
    new_q = generate_random_quaternion(q.shape).to(q.device)
    dot_product = torch.sum(q*new_q, dim=-1, keepdim=True)  #
    shape = (tuple(1 for _ in range(len(dot_product.shape) - 1)) + (4, ))
    q_orthogonal = normalize(new_q - q * dot_product.repeat(*shape))
    # theta = 2arccos(|p.dot(q)|)
    # |p.dot(q)| = cos(theta/2)
    tile_theta = theta.repeat(shape)
    jittered_q = q*torch.cos(tile_theta/2) + q_orthogonal*torch.sin(tile_theta/2)

    return jittered_q


def project_to_axis(q, v):  # [Bs, 4], [Bs, 3]
    a = q[..., 0]  # [Bs]
    b = torch.sum(q[..., 1:] * v, dim=-1)  # [Bs]
    rad = 2 * torch.atan2(b, a)
    new_q = axis_theta_to_quater(v, rad)
    residual = relative_angle(q, new_q)

    return rad, new_q, residual


def rotate_points_with_rotvec(points, rot_vec): # [Bs, 3], [Bs, 3]
    """
    Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    theta = torch.norm(rot_vec, dim=-1, keepdim=True)  # [Bs, 1, 1]
    mask = (theta < 1e-8).float()
    v = rot_vec / torch.max(theta, mask)  # [Bs, 1, 1]
    dot = torch.sum(points * v, dim=-1, keepdim=True)  # [Bs, N, 1]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    return cos_theta * points + sin_theta * torch.cross(v, points, dim=-1) + dot * (1 - cos_theta) * v


def rot_diff_rad(rot1, rot2):
    mat_diff = torch.matmul(rot1, rot2.transpose(-1, -2))
    diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
    diff = (diff - 1) / 2.0
    diff = torch.clamp(diff, min=-1.0, max=1.0)
    return torch.acos(diff)


def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180.0


def generate_random_quaternion(quaternion_shape):
    assert quaternion_shape[-1] == 4
    rand_norm = torch.randn(quaternion_shape)
    rand_q = normalize(rand_norm)
    return rand_q


def noisy_rot_matrix(matrix, rad, type='normal'):
    if type == 'normal':
        theta = torch.abs(torch.randn_like(matrix[..., 0, 0])) * rad
    elif type == 'uniform':
        theta = torch.rand_like(matrix[..., 0, 0]) * rad
    quater = matrix_to_unit_quaternion(matrix)
    new_quater = jitter_quaternion(quater, theta.unsqueeze(-1))
    new_mat = unit_quaternion_to_matrix(new_quater)
    return new_mat


def jitter_rot_matrix(matrix, rad):
    quater = matrix_to_unit_quaternion(matrix)
    new_quater = jitter_quaternion(quater, rad.unsqueeze(-1))
    new_mat = unit_quaternion_to_matrix(new_quater)
    return new_mat


def rotate_around_point(points, rotation, pivot):  # [Bs, N, 3], [Bs, 3, 3], [Bs, 3]
    pivot = pivot.unsqueeze(-2)  # [Bs, 1, 3]
    points = torch.matmul(points - pivot, rotation.transpose(-1, -2))  # [Bs, N, 3], [Bs, 3, 3]
    points = points + pivot
    return points


def normalize_vector(v):  # v: [B, 3]
    batch = v.shape[0]
    v_mag = torch.norm(v, p=2, dim=1)  # batch

    eps = torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v_mag.device))
    valid_mask = (v_mag > eps).float().view(batch, 1)
    backup = torch.tensor([1.0, 0.0, 0.0]).float().to(v.device).view(1, 3).expand(batch, 3)
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    ret = v * valid_mask + backup * (1 - valid_mask)

    return ret


def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def proj_u_a(u, a):
    batch = u.shape[0]
    top = u[:, 0] * a[:, 0] + u[:, 1] * a[:, 1] + u[:, 2] * a[:, 2]
    bottom = u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1] + u[:, 2] * u[:, 2]
    bottom = torch.max(torch.autograd.Variable(torch.zeros(batch).to(bottom.device)) + 1e-8, bottom)
    factor = (top / bottom).view(batch, 1).expand(batch, 3)
    out = factor * u
    return out


def compute_rotation_matrix_from_matrix(matrices):
    b = matrices.shape[0]
    a1 = matrices[:, :, 0]  # batch*3
    a2 = matrices[:, :, 1]
    a3 = matrices[:, :, 2]

    u1 = a1
    u2 = a2 - proj_u_a(u1, a2)
    u3 = a3 - proj_u_a(u1, a3) - proj_u_a(u2, a3)

    e1 = normalize_vector(u1)
    e2 = normalize_vector(u2)
    e3 = normalize_vector(u3)

    rmat = torch.cat((e1.view(b, 3, 1), e2.view(b, 3, 1), e3.view(b, 3, 1)), 2)

    return rmat


def compute_rotation_matrix_from_3d(vec):  # [B, 3]
    y = normalize_vector(vec)  # [B, 3]
    x_raw = torch.zeros_like(y)  # [B, 3]
    x_raw[..., 0] = 1.0
    z = cross_product(x_raw, y)
    z = normalize_vector(z)
    x = cross_product(y, z)

    x = x.view(-1, 3, 1)  # [B, 1, 3]
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), -1)  # [B, 3, 1] --> [B, 3, 3]
    return matrix