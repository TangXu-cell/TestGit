# -*- coding=utf-8 -*-
#利用牛顿法求解非线性方程组，具有二阶收敛速度，对初始迭代值要求很高，所以迭代初值取为第二小问优化后的赤经，赤纬
#星敏感器成像焦距取常用星敏感器(CCD,CMOS)焦距为参考值，成像表面像素尺寸取常用相机尺寸值为参考值
from sympy import *
from numpy.linalg import *
from timeit import Timer
import math
import numpy as np
import matplotlib.pyplot as plt
from Tkinter import *

def NewtonMethod():
        global N
        #三个恒星在坐标系的赤经和赤纬以及在感光面上的星像点质心中心位置
        #1代表用程序给定的值测试，2表示用自定义数据测试
        Test = input('请输入测试指令:')
        if (Test ==1):
                Para1 = [116.54,-37.93,330]
                Para2 = [117.02,-25.93,258]
                Para3 = [119.77,-23.31,192]
                Para4 = [65.7893,-12.5257,0.015]
                f = 36.5352
        elif (Test==2):
                Para1 = list(input('请输入恒星P1的信息:'))
                Para2 = list(input('请输入恒星P2的信息:'))
                Para3 = list(input('请输入恒星P3的信息:'))
                Para4 = list(input('请输入初始信息:'))
                f = input('请输入星敏感器相机焦距:')
        else:
                print '输入指令错误！！'
        alfa1,delta1,a1 = Para1
        alfa2,delta2,a2 = Para2
        alfa3,delta3,a3 = Para3
        pi = math.pi
        #三个方程组求雅可比矩阵
        Jacob1_1,Jacob1_2,Jacob1_3,Jacob2_1,Jacob2_2,Jacob2_3,Jacob3_1,Jacob3_2,Jacob3_3,alfa0,delta0,c=symbols("Jacob1_1 Jacob1_2 Jacob1_3 Jacob2_1 Jacob2_2 Jacob2_3\
                                                                                                                                                                                          Jacob3_1 Jacob3_2 Jacob3_3 alfa0 delta0 c")
                                                                                                                                                                                        
        y1 = (-sin(alfa0*pi/180)*cos(alfa1*pi/180)*cos(delta1*pi/180)+cos(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa1*pi/180)*cos(delta1*pi/180)-cos(alfa0*pi/180)*cos(delta0*pi/180)*sin(delta1*pi/180))**2+\
                       (-cos(alfa0*pi/180)*cos(alfa1*pi/180)*cos(delta1*pi/180)-sin(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa1*pi/180)*cos(delta1*pi/180)-sin(alfa0*pi/180)*cos(delta0*pi/180)\
                        *sin(delta1*pi/180))**2-(c*a1*(-cos(delta0*pi/180)*sin(alfa1*pi/180)*cos(delta1*pi/180)+sin(delta0*pi/180)*sin(delta1*pi/180))/f)**2
        y2,alfa0,delta0,c= symbols("y2 alfa0 delta0 c")
        y2 = (-sin(alfa0*pi/180)*cos(alfa2*pi/180)*cos(delta2*pi/180)+cos(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa2*pi/180)*cos(delta2*pi/180)-cos(alfa0*pi/180)*cos(delta0*pi/180)*sin(delta2*pi/180))**2+\
                       (-cos(alfa0*pi/180)*cos(alfa2*pi/180)*cos(delta2*pi/180)-sin(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa2*pi/180)*cos(delta2*pi/180)-sin(alfa0*pi/180)*cos(delta0*pi/180)\
                        *sin(delta2*pi/180))**2-(c*a2*(-cos(delta0*pi/180)*sin(alfa2*pi/180)*cos(delta2*pi/180)+sin(delta0*pi/180)*sin(delta2*pi/180))/f)**2
        y3,alfa0,delta0,c= symbols("y3 alfa0 delta0 c")
        y3 = (-sin(alfa0*pi/180)*cos(alfa3*pi/180)*cos(delta3*pi/180)+cos(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa3*pi/180)*cos(delta3*pi/180)-cos(alfa0*pi/180)*cos(delta0*pi/180)*sin(delta3*pi/180))**2+\
                       (-cos(alfa0*pi/180)*cos(alfa3*pi/180)*cos(delta3*pi/180)-sin(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa3*pi/180)*cos(delta3*pi/180)-sin(alfa0*pi/180)*cos(delta0*pi/180)\
                        *sin(delta3*pi/180))**2-(c*a3*(-cos(delta0*pi/180)*sin(alfa3*pi/180)*cos(delta3*pi/180)+sin(delta0*pi/180)*sin(delta3*pi/180))/f)**2
        Xk = np.mat(Para4).reshape(3,1)
        #迭代可视化数据库
        Xk_Plot = Xk.reshape(1,3)
        Iteras = [0]
        alfa0_plot = []
        delta0_plot = []
        c_plot = []
        funcs1 =Matrix([y1])
        funcs2 =Matrix([y2])
        funcs3 =Matrix([y3])
        args = Matrix([alfa0,delta0,c])
        res1 = funcs1.jacobian(args)
        res2 = funcs2.jacobian(args)
        res3 = funcs3.jacobian(args)
        #使用牛顿迭代法，预定义迭代N次
        for i in range (1,N+1):
                F_x = [float(y1.evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]})),float(y2.evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]})),float(y3.evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))]
                F_x = np.mat(F_x).reshape(3,1)
                Jacob1_1 = float(np.mat(res1).tolist()[0][0].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
                Jacob1_2 = float(np.mat(res1).tolist()[0][1].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
                Jacob1_3 = float(np.mat(res1).tolist()[0][2].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
                Jacob2_1 = float(np.mat(res2).tolist()[0][0].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
                Jacob2_2 = float(np.mat(res2).tolist()[0][1].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
                Jacob2_3 = float(np.mat(res2).tolist()[0][2].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
                Jacob3_1 = float(np.mat(res3).tolist()[0][0].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
                Jacob3_2 = float(np.mat(res3).tolist()[0][1].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
                Jacob3_3 = float(np.mat(res3).tolist()[0][2].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
                Jacob =  inv(np.mat([Jacob1_1,Jacob1_2,Jacob1_3,Jacob2_1,Jacob2_2,Jacob2_3,Jacob3_1,Jacob3_2,Jacob3_3]).reshape(3,3))
                Xk_1 = Xk-Jacob*F_x
                Xk_Plot = np.vstack((Xk_Plot,Xk_1.reshape(1,3)))
                Iteras.append(i)
                if norm(Xk_1-Xk)<1e-5:
                        sol = [Regulation1(((Xk_1+Xk)/2).tolist()[0][0]),Regulation2(((Xk_1+Xk)/2).tolist()[1][0])]
                        sol.append(abs(((Xk_1+Xk)/2.0).tolist()[2][0]))
                        print '在预定步内第%d步收敛!!!'%(i)
                        print '用牛顿迭代法收敛后得到方程组的解：赤经=%f度,赤纬=%f度,像素尺寸=%f'%(sol[0],sol[1],sol[2])
                        N = i
                        break
                else:
                        Xk = Xk_1
                        Para4[0] = Xk.tolist()[0][0]
                        Para4[1] = Xk.tolist()[1][0]
                        Para4[2] = Xk.tolist()[2][0]
        else:
                sol = [Regulation1(((Xk_1+Xk)/2).tolist()[0][0]),Regulation2(((Xk_1+Xk)/2).tolist()[1][0])]
                sol.append(abs(((Xk_1+Xk)/2).tolist()[2][0]))
                print '用牛顿迭代法得到方程组的近似解：赤经=%f度,赤纬=%f度,像素尺寸=%f'%(sol[0],sol[1],sol[2])
        #迭代过程可视化
        for i in range(Xk_Plot.shape[0]):
                alfa0_plot.append(Xk_Plot[:,0].tolist()[i][0])
                delta0_plot.append(Xk_Plot[:,1].tolist()[i][0])
                c_plot.append(Xk_Plot[:,2].tolist()[i][0])
        plt.rcParams['xtick.direction'] ='in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.plot(Iteras,alfa0_plot,'-k',lw=1,label='Right ascension')
        plt.plot(Iteras,delta0_plot,'-r',lw=1,label='Declination')
        plt.plot(Iteras,c_plot,'-g',lw=1,label='Pixel size')
        plt.xlabel('Iteration numbers')
        plt.ylabel('Iteration values')
        plt.title('The Variation of Solution Value with Iteration Step')
        plt.legend(frameon=False)
        plt.xlim((Iteras[0],Iteras[-1]))
        plt.show()

#DFP拟牛顿算法
def DFP_Newton():
        global N
        #三个恒星在坐标系的赤经和赤纬以及在感光面上的星像点质心中心位置
        #1代表用程序给定的值测试，2表示用自定义数据测试
        Test = input('请输入测试指令:')
        if (Test ==1):
                Para1 = [116.54,-37.93,330]
                Para2 = [117.02,-25.93,258]
                Para3 = [119.77,-23.31,192]
                Para4 = [116.7893,-23.5257,0.015]
                f = 36.5352
        elif (Test==2):
                Para1 = list(input('请输入恒星P1的信息:'))
                Para2 = list(input('请输入恒星P2的信息:'))
                Para3 = list(input('请输入恒星P3的信息:'))
                Para4 = list(input('请输入初始信息:'))
                f = input('请输入星敏感器相机焦距:')
        else:
                print '输入指令错误！！'
        alfa1,delta1,a1 = Para1
        alfa2,delta2,a2 = Para2
        alfa3,delta3,a3 = Para3
        pi = math.pi
        #三个方程组求雅可比矩阵
        Jacob1_1,Jacob1_2,Jacob1_3,Jacob2_1,Jacob2_2,Jacob2_3,Jacob3_1,Jacob3_2,Jacob3_3,alfa0,delta0,c=symbols("Jacob1_1 Jacob1_2 Jacob1_3 Jacob2_1 Jacob2_2 Jacob2_3\
                                                                                                                                                                                          Jacob3_1 Jacob3_2 Jacob3_3 alfa0 delta0 c")
                                                                                                                                                                                        
        y1 = (-sin(alfa0*pi/180)*cos(alfa1*pi/180)*cos(delta1*pi/180)+cos(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa1*pi/180)*cos(delta1*pi/180)-cos(alfa0*pi/180)*cos(delta0*pi/180)*sin(delta1*pi/180))**2+\
                       (-cos(alfa0*pi/180)*cos(alfa1*pi/180)*cos(delta1*pi/180)-sin(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa1*pi/180)*cos(delta1*pi/180)-sin(alfa0*pi/180)*cos(delta0*pi/180)\
                        *sin(delta1*pi/180))**2-(c*a1*(-cos(delta0*pi/180)*sin(alfa1*pi/180)*cos(delta1*pi/180)+sin(delta0*pi/180)*sin(delta1*pi/180))/f)**2
        y2,alfa0,delta0,c= symbols("y2 alfa0 delta0 c")
        y2 = (-sin(alfa0*pi/180)*cos(alfa2*pi/180)*cos(delta2*pi/180)+cos(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa2*pi/180)*cos(delta2*pi/180)-cos(alfa0*pi/180)*cos(delta0*pi/180)*sin(delta2*pi/180))**2+\
                       (-cos(alfa0*pi/180)*cos(alfa2*pi/180)*cos(delta2*pi/180)-sin(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa2*pi/180)*cos(delta2*pi/180)-sin(alfa0*pi/180)*cos(delta0*pi/180)\
                        *sin(delta2*pi/180))**2-(c*a2*(-cos(delta0*pi/180)*sin(alfa2*pi/180)*cos(delta2*pi/180)+sin(delta0*pi/180)*sin(delta2*pi/180))/f)**2
        y3,alfa0,delta0,c= symbols("y3 alfa0 delta0 c")
        y3 = (-sin(alfa0*pi/180)*cos(alfa3*pi/180)*cos(delta3*pi/180)+cos(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa3*pi/180)*cos(delta3*pi/180)-cos(alfa0*pi/180)*cos(delta0*pi/180)*sin(delta3*pi/180))**2+\
                       (-cos(alfa0*pi/180)*cos(alfa3*pi/180)*cos(delta3*pi/180)-sin(alfa0*pi/180)*sin(delta0*pi/180)*sin(alfa3*pi/180)*cos(delta3*pi/180)-sin(alfa0*pi/180)*cos(delta0*pi/180)\
                        *sin(delta3*pi/180))**2-(c*a3*(-cos(delta0*pi/180)*sin(alfa3*pi/180)*cos(delta3*pi/180)+sin(delta0*pi/180)*sin(delta3*pi/180))/f)**2
        Xk = np.mat(Para4).reshape(3,1)
        #迭代可视化数据库
        Xk_Plot = Xk.reshape(1,3)
        Iteras = [0]
        alfa0_plot = []
        delta0_plot = []
        c_plot = []
        funcs1 =Matrix([y1])
        funcs2 =Matrix([y2])
        funcs3 =Matrix([y3])
        args = Matrix([alfa0,delta0,c])
        res1 = funcs1.jacobian(args)
        res2 = funcs2.jacobian(args)
        res3 = funcs3.jacobian(args)
        Jacob1_1 = float(np.mat(res1).tolist()[0][0].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
        Jacob1_2 = float(np.mat(res1).tolist()[0][1].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
        Jacob1_3 = float(np.mat(res1).tolist()[0][2].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
        Jacob2_1 = float(np.mat(res2).tolist()[0][0].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
        Jacob2_2 = float(np.mat(res2).tolist()[0][1].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
        Jacob2_3 = float(np.mat(res2).tolist()[0][2].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
        Jacob3_1 = float(np.mat(res3).tolist()[0][0].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
        Jacob3_2 = float(np.mat(res3).tolist()[0][1].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
        Jacob3_3 = float(np.mat(res3).tolist()[0][2].evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))
        Jacob = [Jacob1_1,Jacob1_2,Jacob1_3,Jacob2_1,Jacob2_2,Jacob2_3,Jacob3_1,Jacob3_2,Jacob3_3]
        #设置F_x初始值
        F_x = np.mat([float(y1.evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]})),float(y2.evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]})),
                       float(y3.evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))]).reshape(3,1)
        #设置雅可比矩阵逆矩阵的初始值
        Jacob =inv(np.mat(Jacob).reshape(3,3))
        Pixel0 = Para4[2]
        #使用DFP拟牛顿迭代法，预定义迭代N次
        for i in range (1,N+1):
                temp = F_x
                Xk_1 = Xk - Jacob*F_x
                Delta_Xk = Xk_1-Xk
                Xk_Plot = np.vstack((Xk_Plot,Xk_1.reshape(1,3)))
                Iteras.append(i)
                #更新F_x
                Xk = Xk_1
                Para4[0] = Xk.tolist()[0][0]
                Para4[1] = Xk.tolist()[1][0]
                Para4[2] = Xk.tolist()[2][0]
                F_x = np.mat([float(y1.evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]})),float(y2.evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]})),
                       float(y3.evalf(subs={alfa0:Para4[0],delta0:Para4[1],c:Para4[2]}))]).reshape(3,1)
                #对比单个多元函数梯度值的增量步，这里表示为函数值向量的增量步
                Delta_Fx = F_x-temp
                Jacob =  Jacob-Jacob*Delta_Fx*Delta_Fx.T*Jacob/(Delta_Fx.T*Jacob*Delta_Fx)+Delta_Xk*Delta_Xk.T/(Delta_Xk.T*Delta_Fx)
                #控制像素尺寸大小
                if 0<Xk[2,0]<0.1:
                        pass
                else:
                        Xk[2,0]  = (exp(Xk[2,0])-exp(-Xk[2,0]))/(exp(Xk[2,0]+exp(-Xk[2,0])))+Pixel0
                #Delta_Fx绝对值的平均
                print F_x
                if norm(F_x,ord=np.inf)/3.0<1e-2:
                        sol = [Regulation1(Xk.tolist()[0][0]),Regulation2(Xk.tolist()[1][0])]
                        sol.append(abs(((Xk_1+Xk)/2.0).tolist()[2][0]))
                        print '在预定步内第%d步收敛!!!'%(i)
                        print '用DFP拟牛顿迭代法收敛后得到方程组的解：赤经=%f度,赤纬=%f度,像素尺寸=%f'%(sol[0],sol[1],sol[2])
                        N = i
                        break
        else:
                sol = [Regulation1(((Xk_1+Xk)/2).tolist()[0][0]),Regulation2(((Xk_1+Xk)/2).tolist()[1][0])]
                sol.append(abs(((Xk_1+Xk)/2).tolist()[2][0]))
                print '用DFP拟牛顿迭代法得到方程组的近似解：赤经=%f度,赤纬=%f度,像素尺寸=%f'%(sol[0],sol[1],sol[2])
        #迭代过程可视化
        for i in range(Xk_Plot.shape[0]):
                alfa0_plot.append(Xk_Plot[:,0].tolist()[i][0])
                delta0_plot.append(Xk_Plot[:,1].tolist()[i][0])
                c_plot.append(Xk_Plot[:,2].tolist()[i][0])
        plt.rcParams['xtick.direction'] ='in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.plot(Iteras,alfa0_plot,'-k',lw=1,label='Right ascension')
        plt.plot(Iteras,delta0_plot,'-r',lw=1,label='Declination')
        plt.plot(Iteras,c_plot,'-g',lw=1,label='Pixel size')
        plt.xlabel('Iteration numbers')
        plt.ylabel('Iteration values')
        plt.title('The Variation of Solution Value with Iteration Step')
        plt.legend(frameon=False)
        plt.xlim((Iteras[0],Iteras[-1]))
        plt.show()
        
        

#赤经角度落在0度到360度范围内
def Regulation1(a):
        while(not(0.0<=a<=360.0)):
                if a>0.0:
                        a = a-360.0
                else:
                        a = a+360.0
        return a

#赤纬的角度落在-90度到90度范围内
def Regulation2(b):
        while(not(-90.0<=b<=90.0)):
                if (-180.0<=b<=180.0):
                        if -180.0<=b<=-90.0:
                                b =180.0+b
                        else:
                                b =180.0-b
                else:
                        if (b<-180.0):
                                b = b+360.0
                        else:
                                b = b-360.0
        return b
                
        
def Call_newtons():
        #初始迭代次数
        N = input('请输入预定迭代步数:')
        #测试程序运行时间
        t1 = Timer("NewtonMethod()","from __main__ import NewtonMethod")
        Newton_Iter = t1.timeit(number=1)
        print '用牛顿迭代算法迭代运行时间：'+str(Newton_Iter)+',迭代速度：'+str(Newton_Iter/N)+' s/步'
        N = input('请输入预定迭代步数:')
        t2 = Timer("DFP_Newton()","from __main__ import DFP_Newton")
        DFP_Iter = t2.timeit(number=1)
        print '用DFP拟牛顿迭代算法迭代运行时间：'+str(DFP_Iter)+',迭代速度：'+str(DFP_Iter/N)+' s/步'

if __name__=='__main__':
        top = Tk()
        top.title('牛顿迭代类算法测试')
        top.geometry("400x400+200+50")
        Button(top,text='牛顿迭代法求解',command=NewtonMethod,width=20,height=5).grid(row=4,column=4,sticky=W,padx=25,pady=60)
        Button(top,text='牛顿迭代法计算时间测试',command=NewtonMethod,width=20,height=5).grid(row=4,column=8,sticky=W,padx=20,pady=60)
        Button(top,text='DFP拟牛顿法求解',command=NewtonMethod,width=20,height=5).grid(row=8,column=4,sticky=W,padx=25,pady=20)
        Button(top,text='DFP计算时间测试',command=NewtonMethod,width=20,height=5).grid(row=8,column=8,sticky=W,padx=20,pady=20)
        top.mainloop()

        
        
        
                               
                        
                               

        
        
                
                
        
        
        
                

                
                
                
                
                
        
        
        
        
        
        
        

        
        
        
        
        


