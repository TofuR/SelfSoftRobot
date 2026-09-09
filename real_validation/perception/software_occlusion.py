"""Explicit software perturbations; no mask is given to the state observer."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class OcclusionConfig:
    enabled: bool = False
    # Source-image percentages: x, y, width, height; grayscale value 0..255.
    rectangles: tuple = ((46.1,46.25,8.75,11.67,35),)

    def __post_init__(self):
        if len(self.rectangles)>16:raise ValueError('最多 16 个遮挡区域')
        for row in self.rectangles:
            if len(row)!=5 or not np.isfinite(row).all():raise ValueError('遮挡参数需为五个有限数值')
            x,y,w,h,g=row
            if min(x,y)<0 or min(w,h)<=0 or x+w>100.001 or y+h>100.001 or not 0<=g<=255:
                raise ValueError('遮挡区域必须位于图像内，宽高大于零，灰度 0..255')

    def apply(self,image):
        result=image.copy()
        if self.enabled:
            height,width=image.shape[:2]
            for x,y,w,h,g in self.rectangles:
                x0,y0=round(x*width/100),round(y*height/100)
                x1,y1=round((x+w)*width/100),round((y+h)*height/100)
                result[y0:y1,x0:x1]=round(g)
        return result,asdict(self)
