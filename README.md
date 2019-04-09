# 项目规则


+ 写文档说明
    + 每个函数写上自己的名字
    + 每个py文件前面写上这个文件主要是干啥的
+ 写测试用例
    + 每个函数至少有一个测试
+ 使用typing约束参数
    + 以便知道传参是否规范

---
文件名|功能
---|---
data|存放数据
data.original_data|存放原始数据，一般不可以修改
data.preocessed_data|存放处理之后的数据，一般是对源文件的一些统计结果，或者是fakedata之类
saved_model|我们训练好的模型都放在这里，最终的仓库，以文件夹的形式，存放最终调好的模型结果，并应有文档说明与predict脚本
test_examples|存放测试用例文件
tmp|一些输出临时可以存放的地方，重要的东西一定要放在saved_model中
tmp.imgs|存放一些图片，例如readme的图片
tmp.results|测试集预测结果放在这里
tmp.model_weight|模型的权重输出到这里
utils|我们的一些可以共用的文件都约定放在这里，要有注释
scripts|公用的，有输入输出的脚本文件，一般是独立完成某项任务
members|开发成员，大家私有的可以自由发挥的文件夹
members.renyan|个人文件夹，自由发挥
members.liangjiaxi|个人文件夹，自由发挥
members.yuzhao|个人文件夹，自由发挥
members.huangmengxuan|个人文件夹，自由发挥


