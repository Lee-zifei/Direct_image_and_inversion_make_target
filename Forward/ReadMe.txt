文件夹和文件说明：

input 文件夹存放速度模型：准确模型命名为acc_vp.bin，代码读入函数为get_acc_model(...)，模型名字没有在参数文件中参数化输入
output 文件夹存放生成的观测数据以及所有的代码输出文件
forward 文件夹为正演代码，参数文件、输入文件夹input、输出文件夹output与成像代码是共享的
parameter.txt 代码参数文件--里面有些参数没有用（useless标注）
Makefile里面的mpicc和nvcc需要更改为自己服务器的路径

*.cpp代码是CPU端执行的代码；
**.cu代码是GPU端执行的并行kernel代码。

执行过程：

先在forward文件夹中make，生成可执行命令，执行生成观测数据（obs-dir=rms表示去掉直达波的反射波数据）；
再make RTM的Makefile，生成偏移可执行命令，执行最终输出偏移结果migration_pp.bin


偏移代码说明：
在RTM.cpp代码的ini_model_mine()函数中window参数表示对准确模型的光滑程度，光滑后的模型即ini_model.bin表示的偏移速度模型（一般做光滑是为了避免层间多次波，只利用primary波《每个成像点的首波》来进行成像）；
观测系统主要是炮点的起始位置sx0和炮间距shotdx，要保证所有炮在模型范围内（0~nx-1)*dx
