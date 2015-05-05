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
from scipy import weave

eps = np.finfo(np.float).eps
    
def chanvese3d(I,init_mask,max_its=200,alpha=0.2,thresh=0,color='r',display=False):

    I = I.astype('float')
    
    #-- Create a signed distance map (SDF) from mask
    phi = mask2phi(init_mask)

    if display:
        plt.ion()
        showCurveAndPhi(I, phi, color)
    
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
                if np.mod(its,10) == 0:            
                    #set(ud.txtInfo1,'string',sprintf('iteration: %d',its),'color',[1 1 0]);
                    print 'iteration:', its
                    showCurveAndPhi(I, phi, color)

            else:
                if np.mod(its,10) == 0:
                    print 'iteration:', its
                    #set(ud.txtInfo1,'string',sprintf('iteration: %d',its),'color',[1 1 0]);
                    #drawnow;

            #-- find interior and exterior mean
            upts = np.flatnonzero(phi<=0)                 # interior points
            vpts = np.flatnonzero(phi>0)                  # exterior points
            u = np.sum(I.flat[upts])/(len(upts)+eps) # interior mean
            v = np.sum(I.flat[vpts])/(len(vpts)+eps) # exterior mean

            F = (I.flat[idx]-u)**2-(I.flat[idx]-v)**2     # force from image information
            curvature = get_curvature(phi,idx)  # force from curvature penalty

            dphidt = F /np.max(np.abs(F)) + alpha*curvature  # gradient descent to minimize energy

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
        time.sleep(10)

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
    # subplot(numRows, numCols, plotNum)
    plt.subplot(321)
    plt.imshow(I[:,:,I.shape[2]/2], cmap='gray')
    plt.hold(True)
    CS = plt.contour(phi[:,:,I.shape[2]/2], 0, colors=color) 
    plt.hold(False)

    plt.subplot(322)
    plt.imshow(phi[:,:,I.shape[2]/2])

    plt.subplot(323)
    plt.imshow(I[:,I.shape[1]/2,:], cmap='gray')
    plt.hold(True)
    CS = plt.contour(phi[:,I.shape[1]/2,:], 0, colors=color) 
    plt.hold(False)

    plt.subplot(324)
    plt.imshow(phi[:,I.shape[1]/2,:])   

    plt.subplot(325)
    plt.imshow(I[I.shape[0]/2,:,:], cmap='gray')
    plt.hold(True)
    CS = plt.contour(phi[I.shape[0]/2,:,:], 0, colors=color) 
    plt.hold(False)

    plt.subplot(326)
    plt.imshow(phi[I.shape[0]/2,:,:])     
    
    plt.draw()
    #time.sleep(1)
    
#-- converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a)-bwdist(1-init_a)+ init_a -0.5
    return phi
  
#-- compute curvature along SDF
def get_curvature(phi,idx):
    dimz, dimy, dimx = phi.shape       
    zyx = np.array([np.unravel_index(i, phi.shape)for i in idx])  # get subscripts
    z = zyx[:,0]
    y = zyx[:,1]
    x = zyx[:,2]
    
    #-- get subscripts of neighbors
    zm1 = z-1; ym1 = y-1; xm1 = x-1;
    zp1 = z+1; yp1 = y+1; xp1 = x+1;

    #-- bounds checking  
    zm1[zm1<0] = 0; ym1[ym1<0] = 0; xm1[xm1<0] = 0;
    zp1[zp1>=dimz]=dimz-1; yp1[yp1>=dimy]=dimy-1; xp1[xp1>=dimx]=dimx-1;

    #-- get central derivatives of SDF at x,y
    dx  = (phi[z,y,xm1]-phi[z,y,xp1])/2 # (l-r)/2
    dxx = phi[z,y,xm1]-2*phi[z,y,x]+phi[z,y,xp1] # l-2c+r
    dx2 = dx*dx

    dy  = (phi[z,ym1,x]-phi[z,yp1,x])/2 # (u-d)/2
    dyy = phi[z,ym1,x]-2*phi[z,y,x]+phi[z,yp1,x] # u-2c+d
    dy2 = dy*dy

    dz  = (phi[zm1,y,x]-phi[zp1,y,x])/2 # (b-f)/2
    dzz = phi[zm1,y,x]-2*phi[z,y,x]+phi[zp1,y,x] # b-2c+f
    dz2 = dz*dz

    # (ul+dr-ur-dl)/4
    dxy = (phi[z,ym1,xm1]+phi[z,yp1,xp1]-phi[z,ym1,xp1]-phi[z,yp1,xm1])/4

    # (lf+rb-rf-lb)/4
    dxz = (phi[zp1,y,xm1]+phi[zm1,y,xp1]-phi[zp1,y,xp1]-phi[zm1,y,xm1])/4

    # (uf+db-df-ub)/4
    dyz = (phi[zp1,ym1,x]+phi[zm1,yp1,x]-phi[zp1,yp1,x]-phi[zm1,ym1,x])/4

    #-- compute curvature (Kappa)
    curvature = ( (dxx*(dy2+dz2)+dyy*(dx2+dz2)+dzz*(dx2+dy2)-
                   2*dx*dy*dxy-2*dx*dz*dxz-2*dy*dz*dyz)/
                  (dx2+dy2+dz2+eps) )

    return curvature

def mymax(a,b):
    return (a + b + np.abs(a - b))/2

#-- level set re-initialization by the sussman method
def sussman(D, dt):
    # forward/backward differences
    a = D - shiftR(D) # backward
    b = shiftL(D) - D # forward
    c = D - shiftD(D) # backward
    d = shiftU(D) - D # forward
    e = D - shiftF(D) # backward
    f = shiftB(D) - D # forward
    
    a_p = a;  a_n = a.copy(); # a+ and a-
    b_p = b;  b_n = b.copy();
    c_p = c;  c_n = c.copy();
    d_p = d;  d_n = d.copy();
    e_p = e;  e_n = e.copy();
    f_p = f;  f_n = f.copy();    

    i_max = D.shape[0]*D.shape[1]*D.shape[2]
    code = """
           for (int i = 0; i < i_max; i++) {
               if ( a_p[i] < 0 ) { a_p[i] = 0; }
               if ( a_n[i] > 0 ) { a_n[i] = 0; }
               if ( b_p[i] < 0 ) { b_p[i] = 0; }
               if ( b_n[i] > 0 ) { b_n[i] = 0; }
               if ( c_p[i] < 0 ) { c_p[i] = 0; }
               if ( c_n[i] > 0 ) { c_n[i] = 0; }
               if ( d_p[i] < 0 ) { d_p[i] = 0; }
               if ( d_n[i] > 0 ) { d_n[i] = 0; }
               if ( e_p[i] < 0 ) { e_p[i] = 0; }
               if ( e_n[i] > 0 ) { e_n[i] = 0; }
               if ( f_p[i] < 0 ) { f_p[i] = 0; }
               if ( f_n[i] > 0 ) { f_n[i] = 0; }
            }
    """
    weave.inline( code,
                  ['i_max',
                   'a_p','a_n','b_p','b_n','c_p','c_n','d_p','d_n','e_p','e_n','f_p','f_n']
                  )
                 
    dD = np.zeros(D.shape)
    D_neg_ind = np.flatnonzero(D < 0)
    D_pos_ind = np.flatnonzero(D > 0)    
       
    dD.flat[D_pos_ind] = np.sqrt( mymax( a_p.flat[D_pos_ind]**2, b_n.flat[D_pos_ind]**2) 
                                  + mymax( c_p.flat[D_pos_ind]**2, d_n.flat[D_pos_ind]**2 )
                                  + mymax( e_p.flat[D_pos_ind]**2, f_n.flat[D_pos_ind]**2 )
                                  ) - 1
    
    dD.flat[D_neg_ind] = np.sqrt( mymax( a_n.flat[D_neg_ind]**2, b_p.flat[D_neg_ind]**2 )
                                  + mymax( c_n.flat[D_neg_ind]**2, d_p.flat[D_neg_ind]**2 )
                                  + mymax( e_n.flat[D_neg_ind]**2, f_p.flat[D_neg_ind]**2 )
                                  ) - 1  
    

    D = D - dt * np.sign(D) * dD
    
    return D
  
#-- whole matrix derivatives
def shiftD(M):
    shift = M[:,range(1,M.shape[1])+[M.shape[1]-1],:]
    return shift

def shiftL(M):
    shift = M[:,:,range(1,M.shape[2])+[M.shape[2]-1]]
    return shift

def shiftR(M):
    shift = M[:,:,[0]+range(0,M.shape[2]-1)]
    return shift

def shiftU(M):
    shift = M[:,[0]+range(0,M.shape[1]-1),:]
    return shift

def shiftF(M):
    shift = M[[0]+range(0,M.shape[0]-1),:,:]
    return shift

def shiftB(M):
    shift = M[range(1,M.shape[0])+[M.shape[0]-1],:,:]
    return shift   

# Convergence Test
def convergence(p_mask,n_mask,thresh,c):
    diff = p_mask - n_mask
    n_diff = np.sum(np.abs(diff))
    if n_diff < thresh:
        c = c + 1
    else:
        c = 0
        
    return c
