
import os,glob
import cv2
import numpy as np

def save_cropped_rotated_faces(fddb_dir,crop_dir,el_dir=None,save_ellipses=False):
    """
    Saves cropped upright faces useful for training and testing.
    Also can save faces with ellipses drawn on them for viewing optionally.
    Assumes directory structure as FDDB.
    """
    assert os.path.isdir(crop_dir)
    if save_ellipses:
        assert os.path.isdir(crop_dir)
    for fold in range(1,11):
        fold_str = "%.02d" %fold
        fold_file_name = os.path.join(fddb_dir,"FDDB-folds","FDDB-fold-{}.txt".format(fold_str))
        fold_annotation_file_name = os.path.join(fddb_dir,"FDDB-folds","FDDB-fold-{}-ellipseList.txt".format(fold_str))
        fold_annotation = open(fold_annotation_file_name,'r').read().split('\n')
        fold_images = open(fold_file_name, 'r').read().split()
        print(fold, len(fold_images))
        for item in fold_images:
            fp = os.path.join(fddb_dir, "{}.jpg".format(item))
            img = cv2.imread(fp)
            h,w,_ = img.shape
            idx = fold_annotation.index(item)
            assert fold_annotation[idx] == item
            num_lines = int(fold_annotation[idx+1])
            for line_no in range(num_lines):
                face_annot = fold_annotation[idx+line_no+2]
                M,m,ang,cx,cy,_ = list(map(float,face_annot.split()))
                out_file = os.path.join(crop_dir, fp.replace('/','__').replace('.', "_face_{}.".format(line_no)))
                calc_x, calc_y = np.linalg.norm([M*np.cos(ang),m*np.sin(ang)]), np.linalg.norm([M*np.sin(ang),m*np.cos(ang)]), 
                a,b,c,d = max(cy-calc_y,0),min(cy+calc_y,h),max(cx-calc_x,0),min(cx+calc_x,w)
                a,b,c,d = list(map(int, [a,b,c,d]))
                face = img[a:b,c:d]
                face_h, face_w,_ = face.shape
                if ang<=0:
                    rot_mat = cv2.getRotationMatrix2D((face_w/2,face_h/2), (np.pi/2+ang)*180/np.pi, 1)
                else:
                    rot_mat = cv2.getRotationMatrix2D((face_w/2,face_h/2), (ang-np.pi/2)*180/np.pi, 1)
                face_rot = cv2.warpAffine(face, rot_mat, (face_w, face_h))
                cv2.imwrite(out_file, face_rot)

                if save_ellipses:
                    out_file = os.path.join(el_dir, fp.replace('/','__').replace('.', "_face_{}.".format(line_no)))
                    center_coordinates = (int(cx), int(cy))
                    axesLength = (int(M), int(m))
                    angle = int(ang*180/np.pi)
                    startAngle = 0
                    endAngle = 360
                    color = (255, 0, 0)
                    thickness = 5
                    img_draw = cv2.ellipse(img.copy(), center_coordinates, axesLength, angle,
                                              startAngle, endAngle, color, thickness)
                    cv2.imwrite(out_file, img_draw)

def select_faces(crop_dir,sel_dir,min_size=100):
    """
    Shortlists faces for training/testing only good resolution are used so that
    when downsizes, still retains features.
    """
    from shutil import copyfile
    for fp in glob.glob(os.path.join(crop_dir,"*.jpg")):
      _,filename=os.path.split(fp)
      img = cv2.imread(fp)
      h,w = img.shape[:2]
      if h>=min_size and w>=min_size:
          copyfile(fp, os.path.join(sel_dir,filename))

def augment_horizontally_inverted(some_glob):
    """
    Augments horizontally inverted images
    """
    for fp in glob.glob(some_glob):
        img = cv2.imread(fp)
        h,w,_=img.shape
        cv2.imwrite(fp.replace(".jpg","_inv.jpg"), img[:,w::-1,:])


def get_dataset_fast(face_glob, non_face_glob, feature_indices):
    """
    Gets selected haar features fast.
    ---------
    Arguments
    ---------
    face_glob [list(str)]: Glob of image paths containing faces
    non_face_glob [list(str)]: Glob of image paths not containing faces
    feature_indices [list(int)]: List containing feature indices that we want to get   
    -------
    Returns
    -------
    data [np.array]: Numpy array of shape (m,162336)
    labels [np.array]: Numpy array of shape (m,) with ones for faces and zeros for non-faces
    n_pos [int]: Number of face images
    n_neg [int]: Number of non-face images
    n_tot [int]: Total number of images
    """
    face_imgs = sorted(glob.glob(face_glob))
    non_face_imgs = sorted(glob.glob(non_face_glob))
    n_pos, n_neg = len(face_imgs), len(non_face_imgs)
    n_tot = n_pos+n_neg
    labels = np.zeros((n_tot,), dtype=int)
    labels[:n_pos] = 1

    data = np.zeros((n_tot, len(feature_indices)))

    for i, fp in enumerate(face_imgs+non_face_imgs):
        print("Processing img ", i)
        img = cv2.imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(24,24))
        img = (img-img.mean())/img.std()
        int_img = get_integral_image(img)
        data[i] = get_selected_haar_features(int_img, feature_indices)

    return data, labels, n_pos, n_neg, n_tot

def get_dataset(face_glob, non_face_glob):
    """
    Gets selected haar features fast.
    ---------
    Arguments
    ---------
    face_glob [list(str)]: Glob of image paths containing faces
    non_face_glob [list(str)]: Glob of image paths not containing faces
    -------
    Returns
    -------
    data [np.array]: Numpy array of shape (m,162336)
    labels [np.array]: Numpy array of shape (m,) with ones for faces and zeros for non-faces
    n_pos [int]: Number of face images
    n_neg [int]: Number of non-face images
    n_tot [int]: Total number of images
    """
    face_imgs = sorted(glob.glob(face_glob))
    non_face_imgs = sorted(glob.glob(non_face_glob))
    n_pos, n_neg = len(face_imgs), len(non_face_imgs)
    n_tot = n_pos+n_neg
    labels = np.zeros((n_tot,), dtype=int)
    labels[:n_pos] = 1

    a,b,c,d,e = 43200,27600,43200,27600,20736
    data = np.zeros((n_tot, a+b+c+d+e))

    for i, fp in enumerate(face_imgs+non_face_imgs):
        print("Processing img ", i)
        img = cv2.imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(24,24))
        img = (img-img.mean())/img.std()
        int_img = get_integral_image(img)
        data[i,:a] = haar_features_A(int_img)
        data[i,a:a+b] = haar_features_B(int_img)
        data[i,a+b:a+b+c] = haar_features_C(int_img)
        data[i,a+b+c:a+b+c+d] = haar_features_D(int_img)
        data[i,a+b+c+d:a+b+c+d+e] = haar_features_E(int_img)


    return data, labels, n_pos, n_neg, n_tot

