#------------------------------------------------------------------------
# Region Based Active Contour Segmentation
#
# seg = region_seg(I,init_mask,max_its,alpha,display)
#
# Inputs: I           2D image
#         init_mask   Initialization (1 = foreground, 0 = bg)
#         max_its     Number of iterations to run segmentation for
#         alpha       (optional) Weight of smoothing term
#                       higer = smoother.  default = 0.2
#         display     (optional) displays intermediate outputs
#                       default = true
#
# Outputs: seg        Final segmentation mask (1=fg, 0=bg)
#
# Description: This code implements the paper: "Active Contours Without
# Edges" By Chan Vese. This is a nice way to segment images whose
# foregrounds and backgrounds are statistically different and homogeneous.
#
# Example:
# img = imread('tire.tif');
# m = zeros(size(img));
# m(33:33+117,44:44+128) = 1;
# seg = region_seg(img,m,500);
#
# Coded by: Shawn Lankton (www.shawnlankton.com)
#------------------------------------------------------------------------

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

eps = np.finfo(np.float).eps
    
def chanvese(I,init_mask,max_its=200,alpha=0.2,thresh=0,color='r',display=False):

    I = I.astype('float')
    
    #-- Create a signed distance map (SDF) from mask
    phi = mask2phi(init_mask)

    if display:
        plt.ion()
        showCurveAndPhi(I, phi, color)
        plt.savefig('levelset_start.png',bbox_inches='tight')
    
    #--main loop
    its = 0
    stop = False
    prev_mask = init_mask
    c = 0
      
    while (its < max_its and not stop):

        # get the curve's narrow band
        idx = np.flatnonzero( np.logical_and( phi <= 1.2, phi >= -1.2) )
        
        if len(idx) > 0:
            #-- intermediate output
            if display:
                if np.mod(its,50) == 0:            
                    print 'iteration:', its
                    showCurveAndPhi(I, phi, color)

            else:
                if np.mod(its,10) == 0:
                    print 'iteration:', its

            #-- find interior and exterior mean
            upts = np.flatnonzero(phi<=0)                 # interior points
            vpts = np.flatnonzero(phi>0)                  # exterior points
            u = np.sum(I.flat[upts])/(len(upts)+eps) # interior mean
            v = np.sum(I.flat[vpts])/(len(vpts)+eps) # exterior mean

            F = (I.flat[idx]-u)**2-(I.flat[idx]-v)**2    # force from image information
            curvature = get_curvature(phi,idx)  # force from curvature penalty

            dphidt = F /np.max(np.abs(F)) + alpha*curvature # gradient descent to minimize energy

            #-- maintain the CFL condition
            dt = 0.45/(np.max(np.abs(dphidt))+eps)

            #-- evolve the curve
            phi.flat[idx] += dt*dphidt

            #-- Keep SDF smooth
            phi = sussman(phi, 0.5)
            
            new_mask = phi<=0
            c = convergence(prev_mask,new_mask,thresh,c)

            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else: stop = True

        else:
            break

    #-- final output
    if display:
        showCurveAndPhi(I, phi, color)
        plt.savefig('levelset_end.png',bbox_inches='tight')         

    #-- make mask from SDF
    seg = phi<=0 #-- Get mask from levelset
  
    return seg,phi,its


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#-- AUXILIARY FUNCTIONS ----------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

def bwdist(a):
    """ 
    this is an intermediary function, 'a' has only True, False vals, 
    so we convert them into 0, 1 values -- in reverse. True is 0, 
    False is 1, distance_transform_edt wants it that way.
    """
    return nd.distance_transform_edt(a == 0)

import time

#-- Displays the image with curve superimposed
def showCurveAndPhi(I, phi, color):
    myplot = plt.subplot(121)
    myplot.cla()
    axes = myplot.axes
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
    plt.imshow(I, cmap='gray')
    CS = plt.contour(phi, 0, colors=color) 
    plt.draw()

    myplot = plt.subplot(122)
    axes = myplot.axes
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)       
    plt.imshow(phi)

    plt.draw()
  
def im2double(a):
    a = a.astype('float')
    a /= a.max()
    return a
    
#-- converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a)-bwdist(1-init_a)+im2double(init_a) -0.5
    return phi
  
#-- compute curvature along SDF
def get_curvature(phi,idx):
    dimy, dimx = phi.shape       
    yx = np.array([np.unravel_index(i, phi.shape)for i in idx])  # get subscripts
    y = yx[:,0]
    x = yx[:,1]

    #-- get subscripts of neighbors
    ym1 = y-1; xm1 = x-1; yp1 = y+1; xp1 = x+1;

    #-- bounds checking  
    ym1[ym1<0] = 0; xm1[xm1<0] = 0;              
    yp1[yp1>=dimy]=dimy - 1; xp1[xp1>=dimx] = dimx - 1;    

    #-- get indexes for 8 neighbors
    idup = np.ravel_multi_index( (yp1,x),phi.shape)
    iddn = np.ravel_multi_index( (ym1,x),phi.shape)
    idlt = np.ravel_multi_index( (y,xm1),phi.shape)
    idrt = np.ravel_multi_index( (y,xp1),phi.shape)
    idul = np.ravel_multi_index( (yp1,xm1),phi.shape)
    idur = np.ravel_multi_index( (yp1,xp1),phi.shape)
    iddl = np.ravel_multi_index( (ym1,xm1),phi.shape)
    iddr = np.ravel_multi_index( (ym1,xp1),phi.shape)
    
    #-- get central derivatives of SDF at x,y
    phi_x  = -phi.flat[idlt]+phi.flat[idrt]
    phi_y  = -phi.flat[iddn]+phi.flat[idup]
    phi_xx = phi.flat[idlt]-2*phi.flat[idx]+phi.flat[idrt]
    phi_yy = phi.flat[iddn]-2*phi.flat[idx]+phi.flat[idup]
    phi_xy = (-0.25*phi.flat[iddl]-0.25*phi.flat[idur]
               +0.25*phi.flat[iddr]+0.25*phi.flat[idul])
    phi_x2 = phi_x**2
    phi_y2 = phi_y**2
    
    #-- compute curvature (Kappa)
    curvature = ( ((phi_x2*phi_yy + phi_y2*phi_xx - 2*phi_x*phi_y*phi_xy)
                   / (phi_x2 + phi_y2 +eps)**(3/2))
                  *(phi_x2 + phi_y2)**(1/2))

    return curvature

#-- level set re-initialization by the sussman method
def sussman(D, dt):
    # forward/backward differences
    a = D - shiftR(D) # backward
    b = shiftL(D) - D # forward
    c = D - shiftD(D) # backward
    d = shiftU(D) - D # forward

    a_p = a.copy();  a_n = a.copy(); # a+ and a-
    b_p = b.copy();  b_n = b.copy();
    c_p = c.copy();  c_n = c.copy();
    d_p = d.copy();  d_n = d.copy();
    
    a_p[a < 0] = 0
    a_n[a > 0] = 0
    b_p[b < 0] = 0
    b_n[b > 0] = 0
    c_p[c < 0] = 0
    c_n[c > 0] = 0
    d_p[d < 0] = 0
    d_n[d > 0] = 0

    dD = np.zeros(D.shape)
    D_neg_ind = np.flatnonzero(D < 0)
    D_pos_ind = np.flatnonzero(D > 0)
    
    dD.flat[D_pos_ind] = np.sqrt( np.max( np.concatenate( ([a_p.flat[D_pos_ind]**2],
                                                           [b_n.flat[D_pos_ind]**2]) ),
                                          axis=0
                                          )
                                  + np.max( np.concatenate( ([c_p.flat[D_pos_ind]**2],
                                                             [d_n.flat[D_pos_ind]**2])),
                                            axis=0
                                            )
                                  ) - 1
    dD.flat[D_neg_ind] = np.sqrt( np.max( np.concatenate( ([a_n.flat[D_neg_ind]**2],
                                                           [b_p.flat[D_neg_ind]**2])),
                                          axis=0
                                          )
                                  + np.max( np.concatenate( ([c_n.flat[D_neg_ind]**2],
                                                             [d_p.flat[D_neg_ind]**2]) ),
                                            axis=0
                                            )
                                  ) - 1

    D = D - dt * sussman_sign(D) * dD
    
    return D
  
#-- whole matrix derivatives
def shiftD(M):
    return shiftR(M.transpose()).transpose()

def shiftL(M):
    shift = M[:,range(1,M.shape[1])+[M.shape[1]-1]]
    return shift

def shiftR(M):
    shift = M[:,[0]+range(0,M.shape[1]-1)]
    return shift

def shiftU(M):
    return shiftL(M.transpose()).transpose()
  
def sussman_sign(D):
    return D / np.sqrt(D**2 + 1)    

# Convergence Test
def convergence(p_mask,n_mask,thresh,c):
    diff = p_mask - n_mask
    n_diff = np.sum(np.abs(diff))
    if n_diff < thresh:
        c = c + 1
    else:
        c = 0
        
    return c

if __name__ == "__main__":
    img = nd.imread('brain.png', flatten=True)
    mask = np.zeros(img.shape)
    mask[20:100,20:100] = 1
    
    chanvese(img,mask,max_its=1000,display=True,alpha=1.0)
