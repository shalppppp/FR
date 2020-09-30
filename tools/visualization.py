import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals


# 用matplot进行可视化
def PointCloudVisualization(PointCloud, color = 'r', marker = '.'):
    PointCloud = PointCloud.reshape((-1, 3))
    x=PointCloud[:,0]
    y=PointCloud[:,1]
    z=PointCloud[:,2]
    # print(x)
    fig=plt.figure(dpi=120)
    ax=fig.add_subplot(111,projection='3d')
    plt.title('point cloud')
    ax.scatter(x,y,z,c=color,marker=marker,s=2,linewidth=0,alpha=1,cmap='spectral')

    #ax.set_facecolor((0,0,0))
    # ax.axis('scaled')
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
def CurveVisualization(PointCloud, color = 'r', marker = '.'):
    PointCloud = PointCloud.reshape((-1, 3))
    x=PointCloud[:,0]
    y=PointCloud[:,1]
    z=PointCloud[:,2]
    # print(x)
    fig=plt.figure(dpi=120)
    ax=fig.add_subplot(111,projection='3d')
    plt.title('point cloud')
    ax.plot(x,y,z,c=color)

    #ax.set_facecolor((0,0,0))
    # ax.axis('scaled')
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    ax.set_xlim(-50, 50)
    ax.set_ylabel('')
    ax.set_ylim(-50, 50)
    ax.set_zlabel('')
    ax.set_zlim(-50, 50)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.axis("off")
    plt.show()

def PointCloudVisualbyVispy(data, size=1):
    # def vis_show(data, size = 1):
    data = data.reshape((-1, 3))
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()
    scatter.set_data(data, edge_color='black', face_color=(1, 1, 1, .5), size=size)

    view.add(scatter)

    view.camera = 'turntable'  # or try 'arcball'

    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()

def visualtwodata(data1, data2, size1 = 1, size2 = 3):
    data1 = data1.reshape((-1, 3))
    data2 = data2.reshape((-1, 3))
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()

    org_points = visuals.Markers()
    org_points.set_data(data1, edge_color='black', face_color=(1, 1, 1, .5), size=size1)

    inter_points = visuals.Markers()
    inter_points.set_data(data2, edge_color='red', face_color=(1, 1, 1, .5), size=size2)
    view.add(org_points)
    view.add(inter_points)
    # utils.PointCloudVisualization(data)
    view.camera = 'arcball'
    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()