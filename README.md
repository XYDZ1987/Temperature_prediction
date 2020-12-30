1.各层的参数不是随机初始化的嘛，我用`torch.save(model.state_dict(), 'params1.pth')`保存了一次效果比较好的参数，然后用`model.load_state_dict(torch.load('params1.pth'))`加载参数，加载后别的代码我没有修改，模型会在原来的基础上继续梯度下降。但是每次最初的loss和最终的loss还是不一样（把后面梯度下降的部分注释掉也不行，初始loss还是每次不一样），我开始以为是`Dataloader`把数据打乱了，后面发现不是这个的原因，这是因为啥呀。

2.最近再看李航的《统计学习方法》，里面有几个基础概念我没搞明白。

<img src="./img/pic7.jpg" style="zoom:50%;" /> 

<img src="./img/pic8.jpg" style="zoom:50%;" />

<img src="./img/pic9.jpg" style="zoom:80%;" />

就是这统计学习方法的三要素，我拍了几张书上的图片，csdn上面直接搜统计学习方法三要素也行。我没理解这个“模型”到底是啥意思