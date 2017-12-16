import dlib
import numpy as np
import scipy.misc


f_iden = dlib.get_frontal_face_detector()

pos_68 = dlib.shape_predictor("sixtyEigthFace.dat")

pos_5 = dlib.shape_predictor("fiveFace.dat")

c_f_iden = dlib.cnn_face_detection_model_v1("cnnFace.dat")

f_encr = dlib.face_recognition_model_v1("faceEncode.dat")

 

def r2c(py):
    
    return py.top(), py.right(), py.bottom(), py.left()

def t_f_mark(f_img, fpos=None, mdl="large"):
    if fpos is None:
        fpos= tempfacepos(f_img)
    else:
        fpos= [c2r(facepos) for facepos in fpos]

    pospre = pos_68

    if mdl== "small":
        pospre = pos_5

    return [pospre(f_img, facepos) for facepos in fpos]



def tc2b(c, img):

    return max(c[0], 0), min(c[1], img[1]), min(c[2], img[0]), max(c[3], 0)


def f_disp(f_ecs, fcomp):
    
    if len(f_ecs) == 0:
        return np.empty((0))

    return np.linalg.norm(f_ecs - fcomp, axis=1)



def tempfacepos(img, sample=1, md="hog"):
   
    if md== "cnn":
        return c_f_iden(img, sample)
    else:
        return f_iden(img, sample)

def disgface(gen_f_ecs, inspect_f_ecs, allowance=0.6):
   
    return list(f_disp(gen_f_ecs, inspect_f_ecs) <= allowance)
  
  
  
def loadImageFile(file, mode='RGB'):
    
    return scipy.misc.imread(file, mode=mode)

  
def facepos(img, sample=1, mdl="hog"):
    
    if mdl== "cnn":
        return [tc2b(r2c(profile.rect), img.shape) for profile in tempfacepos(img, sample, "cnn")]
    else:
        return [tc2b(r2c(profile), img.shape) for profile in tempfacepos(img, sample, mdl)]

def f_mark(face_image, face_locations=None):
    landmarks = t_f_mark(face_image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
} for points in landmarks_as_tuples]




def f_ecs(f_map, gen_f_pos=None, nvar=1):
    
    tempmarks = t_f_mark(f_map, gen_f_pos, mdl="small")

    return [np.array(f_encr.compute_face_descriptor(f_map, tempmarkset, nvar)) for tempmarkset in tempmarks]




def c2r(c):
    
    return dlib.rectangle(c[3], c[0], c[1], c[2])

