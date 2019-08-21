import numpy as np
import random

def ifftnc(x, axes):
    tmp = np.fft.fftshift(x, axes=axes)
    tmp = np.fft.ifftn(tmp, axes=axes)
    return np.fft.ifftshift(tmp, axes=axes)

def fftnc(x, axes):
    tmp = np.fft.fftshift(x, axes=axes)
    tmp = np.fft.fftn(tmp, axes=axes)
    return np.fft.ifftshift(tmp, axes=axes)

def fftc(x, axis=0, do_orthonorm=True):
    if do_orthonorm:
        scale = np.sqrt(x.shape[axis])
    else:
        scale = 1.0
    return fftnc(x, (axis,)) / scale

def ifftc(x, axis=0, do_orthonorm=True):
    if do_orthonorm:
        scale = np.sqrt(x.shape[axis])
    else:
        scale = 1.0
    return ifftnc(x, (axis,)) * scale

def fft2c(x, order='C', do_orthonorm=True):
    if order == 'C':
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[-2:]))
        else:
            scale = 1.0
        return fftnc(x, (-2, -1)) / scale
    else:
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[:2]))
        else:
            scale = 1.0
        return fftnc(x, (0, 1)) / scale

def ifft2c(x, order='C', do_orthonorm=True):
    if order == 'C':
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[-2:]))
        else:
            scale = 1.0
        return ifftnc(x, (-2, -1)) * scale
    else:
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[:2]))
        else:
            scale = 1.0
        return ifftnc(x, (0, 1)) * scale

def generate_random_range(start, end, artifact_length=[0,20], gap_length=[0,100], start_ofset=[0,20]):
    
    list = []
    
    current = start + random.randint(start_ofset[0], start_ofset[1])
    
    while current < end:
        
        increment = random.randint(artifact_length[0], artifact_length[1])
        
        if (current + increment > end):
            list.append([current, end])
            return list
        else:
            list.append([current, current + increment])
        
        current = current + increment + random.randint(gap_length[0], gap_length[1])
        
    return list
        


def add_motion_artifact(im, seed=777):
    
    random.seed(a=seed)
    
    im_transformed = fft2c(im)
    vals = generate_random_range(0, 511, artifact_length=[10,60], gap_length=[50,120], start_ofset=[0,50])
    print(vals)
    
    im_transformed_artifacted = add_artifact(im_transformed, regions=vals, max_lf=1200)

    image = reconstruct(im_transformed_artifacted)
    image = np.uint8(image)
    
    return image
    
def row_artifact(row, linear_factor):
        
    alpha = random.uniform(-1 * linear_factor, linear_factor)
    
    counter = 0
    for t in np.linspace(0, 0.002, num=row.size):
        row[counter] = row[counter] * np.exp(alpha * t )
        
        counter = counter + 1
        
    return row

def add_artifact(img, regions=[[0,511]], max_lf=500, seed=777): 
    
    random.seed(seed)
    
    for region in regions:
        
        for index in range(region[0], region[1]):
            img[index] = row_artifact(img[index], max_lf)
        
    return img

def reconstruct(img):
    
    img = np.absolute(ifft2c(img))

    img = np.around(img * 255.0 / np.max(img))
    
    return img

