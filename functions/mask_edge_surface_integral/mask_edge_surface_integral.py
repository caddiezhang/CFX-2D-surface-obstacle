import numpy as np


def edge_finder(mask, px_offset):
    """
    Find the boundaries of mask regions and FOV.

    :param mask: 2D array, masked regions labelled by 1, solved region labelled by 0
    :param px_offset: number of pixels away from the surface
    :return:
    > idx_mask: row and column id of mask regions
    > idx_fov_left, idx_fov_top, idx_fov_right, idx_fov_bottom:
      /row and column id of left, top, right, bottom FOV boundaries excluding the overlapping mask regions
    > idx_fov: row and column id of all FOV boundaries excluding the overlapping mask regions
    > idx_left, idx_top, idx_right, idx_bottom:
      /row and column id of left, top, right, bottom mask boundaries
    > idx_nanbd: row and column id of all mask boundaries

    """

    nrow, ncol = np.shape(mask)
    '''
    find the entire mask region
    '''

    idx_mask = np.where(mask == 1)

    '''
    find the boundaries of FOV
    '''
    idx_mask_pointer = idx_mask[0] + nrow * idx_mask[1]
    idx_fov_left_pointer = list(range(nrow))
    idx_fov_top_pointer = list(np.arange(1, ncol - 1) * nrow)
    idx_fov_right_pointer = list(np.arange(nrow) + (ncol - 1) * nrow)
    idx_fov_bottom_pointer = list(np.arange(1, ncol - 1) * nrow + nrow - 1)
    # remove mask regions from FOV boundaries
    for idl in idx_fov_left_pointer:
        if idl in idx_mask_pointer:
            idx_fov_left_pointer.remove(idl)
    for idt in idx_fov_top_pointer:
        if idt in idx_mask_pointer:
            idx_fov_top_pointer.remove(idt)
    for idr in idx_fov_right_pointer:
        if idr in idx_mask_pointer:
            idx_fov_right_pointer.remove(idr)
    for idb in idx_fov_bottom_pointer:
        if idb in idx_mask_pointer:
            idx_fov_bottom_pointer.remove(idb)
    idx_fov_left = list()
    idx_fov_left.append(np.remainder(idx_fov_left_pointer, nrow).astype('int64'))
    idx_fov_left.append(np.floor(np.array(idx_fov_left_pointer) / nrow).astype('int64'))
    idx_fov_top = list()
    idx_fov_top.append(np.remainder(idx_fov_top_pointer, nrow).astype('int64'))
    idx_fov_top.append(np.floor(np.array(idx_fov_top_pointer) / nrow).astype('int64'))
    idx_fov_right = list()
    idx_fov_right.append(np.remainder(idx_fov_right_pointer, nrow).astype('int64'))
    idx_fov_right.append(np.floor(np.array(idx_fov_right_pointer) / nrow).astype('int64'))
    idx_fov_bottom = list()
    idx_fov_bottom.append(np.remainder(idx_fov_bottom_pointer, nrow).astype('int64'))
    idx_fov_bottom.append(np.floor(np.array(idx_fov_bottom_pointer) / nrow).astype('int64'))
    del idx_fov_left_pointer, idx_fov_top_pointer, idx_fov_right_pointer, idx_fov_bottom_pointer
    # all boundaries of the FOV
    idx_fov = list()
    idx_fov.append(np.concatenate((idx_fov_left[0], idx_fov_top[0], idx_fov_right[0], idx_fov_bottom[0]), axis=0))
    idx_fov.append(np.concatenate((idx_fov_left[1], idx_fov_top[1], idx_fov_right[1], idx_fov_bottom[1]), axis=0))
    '''
    find mask boundaries
    '''
    # find mask left boundary
    idx_left = list()
    idx_left.append(np.where(np.logical_and(mask[:, 1:ncol - 1] == 1, mask[:, 0:ncol - 2] == 0))[0])
    idx_left.append(np.array(np.where(np.logical_and(mask[:, 1:ncol - 1] == 1, mask[:, 0:ncol - 2] == 0))[1])-px_offset)
    # find mask top boundary
    idx_top = list()
    idx_top.append(np.array(np.where(np.logical_and(mask[1:nrow - 1, :] == 1, mask[0:nrow - 2, :] == 0))[0])-px_offset)
    idx_top.append(np.where(np.logical_and(mask[1:nrow - 1, :] == 1, mask[0:nrow - 2, :] == 0))[1])
    # find mask right boundary
    idx_right = list()
    idx_right.append(np.where(np.logical_and(mask[:, 0:ncol - 2] == 1, mask[:, 1:ncol - 1] == 0))[0])
    idx_right.append(np.array(np.where(np.logical_and(mask[:, 0:ncol - 2] == 1, mask[:, 1:ncol - 1] == 0))[1]) + 1 + px_offset)
    # find mask bottom boundary
    idx_bottom = list()
    idx_bottom.append(np.array(np.where(np.logical_and(mask[0:nrow - 2, :] == 1, mask[1:nrow - 1, :] == 0))[0]) + 1 + px_offset)
    idx_bottom.append(np.where(np.logical_and(mask[0:nrow - 2, :] == 1, mask[1:nrow - 1, :] == 0))[1])

    # all boundaries of the mask regions
    idx_nanbd = list()
    idx_nanbd.append(np.concatenate((idx_left[0], idx_top[0], idx_right[0], idx_bottom[0]), axis=0))
    idx_nanbd.append(np.concatenate((idx_left[1], idx_top[1], idx_right[1], idx_bottom[1]), axis=0))

    return idx_mask, \
           idx_fov_left, idx_fov_top, idx_fov_right, idx_fov_bottom, idx_fov, \
           idx_left, idx_top, idx_right, idx_bottom, idx_nanbd
