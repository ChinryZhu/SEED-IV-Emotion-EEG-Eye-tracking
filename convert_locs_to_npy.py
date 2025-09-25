'''

convert_locs_to_npy - Convert the electrode coordinates provided in the SEED-IV dataset into 3D and 2D coordinate format
Copyright (C) 2025 - Chinry Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''


from Utils_Bashivan import *
data_3D=[]
data_2D=[]
with open("Input\\channel_62_pos.locs",'r') as f:
    for line in f:
        num, azimuth_deg, elevation_sin, name=line.split()
        z=float(elevation_sin)
        azimuth_rad = np.deg2rad(float(azimuth_deg))
        xy_scale = np.sqrt(1 - z ** 2)

        x = np.cos(azimuth_rad) * xy_scale
        y = np.sin(azimuth_rad) * xy_scale

        print(x,y,z)
        data_3D.append([x, y, z])
        data_2D.append(azim_proj([x, y, z]))

coord_array_3D = np.array(data_3D, dtype=np.float32)
coord_array_2D = np.array(data_2D, dtype=np.float32)
np.save("Input\\SEED_IV_POS.npy", coord_array_3D)
np.save("Input\\SEED_IV_POS_2D.npy", coord_array_2D)

