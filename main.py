import numpy as np
import cv2
import glob
from skimage.metrics import structural_similarity as compare_ssim

sift = cv2.SIFT_create() # opencv 3
# use "sift = cv2.SIFT()" if the above fails

I2 = cv2.imread('2.jpg')

G2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
keypoints2, desc2 = sift.detectAndCompute(G2, None); # opencv 3

fnames = glob.glob('*.jpg')
fnames.sort()
for fname in fnames:
    
    I2 = cv2.imread('2.jpg')
    G2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
    
    I1 = cv2.imread(fname)
    G1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
    keypoints1, desc1 = sift.detectAndCompute(G1, None); # opencv 3

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2, k=2)
    good_matches = []
    alpha = 0.75
    for m1,m2 in matches:
        # m1 is the best match
        # m2 is the second best match
        if m1.distance < alpha *m2.distance:
            good_matches.append(m1)
    
    if(len(good_matches)) < 4:
        continue

    points2 = [keypoints2[m.trainIdx].pt for m in good_matches]
    points2 = np.array(points2,dtype=np.float32)
    points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
    points1 = np.array(points1,dtype=np.float32)
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0) # this needs to be changed!!
    
    J_origin = cv2.warpPerspective(I1, H, (I2.shape[1],I2.shape[0]) )
    J = cv2.warpPerspective(I1, H, (I2.shape[1],I2.shape[0]) )
    for k in range(J.shape[2]):
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                if J[i,j,k] == 0:
                    J[i,j,k] = I2[i,j,k]
    
    grayA = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(J,cv2.COLOR_BGR2GRAY)
    score, diff = compare_ssim(grayA,grayB,full = True)
    diff = (diff * 255).astype('uint8')
    print(score)
    _, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    if score <= 0.9 and np.sum(H) > 0:
        for cnt in contours:
        
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 1000:
                #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
            
                cv2.rectangle(I2, (x, y), (x + w, y + h), (0,0,255), 5)
    
    # alternatingly show images I2 and J
    ind = 0;
    imgs = [I2, J_origin]
    cv2.imshow('input',cv2.resize(I1, (400,600), interpolation = cv2.INTER_AREA))
    while 1:
        ind = 1-ind

        cv2.imshow('Reg',cv2.resize(imgs[ind], (400,600), interpolation = cv2.INTER_AREA))
        key =  cv2.waitKey(500) 
                
        if key & 0xFF == ord('q'):
            exit()
        elif key & 0xFF != 0xFF:
            break

cv2.destroyAllWindows()
        
