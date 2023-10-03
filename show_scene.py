
# def viz(scene_points, pc_color):
        #     import pyvista as pv
        #     import matplotlib as plt
        #     point_cloud = pv.PolyData(scene_points[:, :3])
        #     point_cloud["elevation"] = pc_color
        #     cmap = plt.cm.get_cmap("jet")
        #     focal_point = (33.595, 3.232, -5.008)
        #     position = (-38.798, 0.306, 25.143)
        #     viewup = (0.3834, 0.026, 0.923)
        #     cpos = [position, focal_point, viewup]
        #     point_cloud.plot(cmap=cmap, window_size=[1600, 800], return_cpos=True, cpos=cpos)
        #     return