import glob,os,cv2,numpy as np
from PIL import Image,ImageEnhance
fs=sorted(glob.glob("decoded/*.tiff"))
assert len(fs)>=2, f"need >=2 frames, got {len(fs)}"
def read(p):
    im=cv2.imread(p,cv2.IMREAD_UNCHANGED)
    if im is None: raise RuntimeError(p)
    if im.shape[2]==4: im=im[:,:,:3]
    return im
ref=read(fs[len(fs)//2])
scale=0.25
rg=cv2.cvtColor(cv2.resize(ref,None,fx=scale,fy=scale),cv2.COLOR_BGR2GRAY).astype(np.float32)
rg=cv2.normalize(rg,None,0,1,cv2.NORM_MINMAX)
warps=[]; good=[]
for p in fs:
    im=read(p)
    g=cv2.cvtColor(cv2.resize(im,None,fx=scale,fy=scale),cv2.COLOR_BGR2GRAY).astype(np.float32)
    g=cv2.normalize(g,None,0,1,cv2.NORM_MINMAX)
    w=np.eye(2,3,dtype=np.float32)
    try:
        cc,w=cv2.findTransformECC(rg,g,w,cv2.MOTION_EUCLIDEAN,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-6),None,3)
        w[:,2]/=scale
        warps.append(w); good.append(p); print(os.path.basename(p),"ECC",cc)
    except cv2.error as e: print("SKIP",p,e)
assert len(good)>=5
H,W=ref.shape[:2]
# Memory-safe robust mean: align each frame, accumulate float64; clipping pass around preliminary mean.
acc=np.zeros((H,W,3),np.float64)
for p,w in zip(good,warps):
    x=read(p).astype(np.float32)
    x=cv2.warpAffine(x,w,(W,H),flags=cv2.INTER_LANCZOS4|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
    acc+=x
mean=(acc/len(good)).astype(np.float32)
# second pass winsorize strong transient outliers
acc[:]=0
for p,w in zip(good,warps):
    x=read(p).astype(np.float32)
    x=cv2.warpAffine(x,w,(W,H),flags=cv2.INTER_LANCZOS4|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
    lo=mean*0.45; hi=np.minimum(mean*2.2+512,65535)
    acc+=np.clip(x,lo,hi)
stack=np.clip(acc/len(good),0,65535).astype(np.uint16)
os.makedirs("output",exist_ok=True)
cv2.imwrite("output/starstack_16bit.tiff",stack)
# display render
rgb=cv2.cvtColor(stack,cv2.COLOR_BGR2RGB).astype(np.float32)/65535
p=np.percentile(rgb,99.8); rgb=np.clip(rgb/max(p,1e-5),0,1)
rgb=np.power(rgb,1/2.2)
im=Image.fromarray((rgb*255).astype(np.uint8))
im.save("output/starstack_natural.jpg",quality=96)
enh=ImageEnhance.Contrast(im).enhance(1.18)
enh=ImageEnhance.Color(enh).enhance(1.12)
enh.save("output/starstack_detail.jpg",quality=96)
print("stacked",len(good),"frames",W,H)