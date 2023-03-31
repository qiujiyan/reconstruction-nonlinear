import numpy as np

class Monitor():
    def __init__(self, 
                 monitorType ,
                 data_shape,
                 random_num=None, 
                 monitorGridShape=None,
                 mask=None
                ):
        self.monitorType = monitorType
        if self.monitorType == "grid":
            if monitorGridShape is None:
                self.monitorGridShape =  (7,7)
            else:
                self.monitorGridShape = monitorGridShape

        if self.monitorType == "random":
            if random_num is None:
                self.random_num = 20
            else:
                self.random_num = random_num
            
        if self.monitorType == "random,sea":
            if random_num is None:
                self.random_num = 20
            else:
                self.random_num = random_num
            if mask is None:
                raise "mask is None"
            self.mask = mask


        self.data_shape = data_shape

    def get_xy(self):
        if self.monitorType == "grid":
            return self.xy_grid()
        if self.monitorType in "random":
            fullShape = self.data_shape
            x = np.random.randint(0,fullShape[0],self.random_num)
            y = np.random.randint(0,fullShape[1],self.random_num)
            self.xy = (x,y)
            return x,y
        if self.monitorType == "random,sea":
            mask = self.mask
            fullShape = self.data_shape
            xy=np.array(np.where(mask==0)).T
            np.random.shuffle(xy)
            xy=xy[:self.random_num]
            xy=xy.T
            x,y = xy
            self.xy = (x,y)
            return x,y
            
    def xy_grid(self):
        monitorShape = self.monitorGridShape 
        fullShape = self.data_shape

        y = np.linspace(0,fullShape[1]-1,monitorShape[1]+2)[1:-1]
        x = np.linspace(0,fullShape[0]-1,monitorShape[0]+2)[1:-1]
        xy_grid = np.meshgrid(x,y)
        xy_grid = np.rint(xy_grid).astype(np.int64)
        return xy_grid
    
    
    def grid2D(self,reqxy=False):
        fullShape = self.data_shape
        gappy_map = np.zeros(fullShape)
        x_grid,y_grid = self.get_xy()
        gappy_map[(x_grid,y_grid)] = 1
        if reqxy:
            return gappy_map,x_grid,y_grid
        else:
            return gappy_map

    def grid1D(self):
        return self.grid2D().flatten()
