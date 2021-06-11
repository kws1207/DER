import pickle
import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import spline
import scipy.interpolate
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

x = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []
y7 = []
y8 = []
y9 = []
y10 = []


t = pickle.load(open('time_topic_dict', 'r'))
time_line_dict = [(k, t[k]) for k in sorted(t.keys())]

for k,v in time_line_dict:
    if int(k) >= 200801:
        x.append(int(k))
        y1.append(float(v[0]))
        y2.append(float(v[1]))
        y3.append(float(v[2]))
        y4.append(float(v[3]))
        y5.append(float(v[4]))



'''

t_user = pickle.load(open('user_topic_dict', 'r'))

for k,v in t_user.items():
    if len(v) > 20:
        dict = {i[0]: i[1] for i in v}
        dict = [(k, dict[k]) for k in sorted(dict.keys())]
        for entry in dict:
            x.append(int(entry[0]))
            y1.append(float(entry[1][0]))
            y2.append(float(entry[1][1]))
            y3.append(float(entry[1][2]))
            y4.append(float(entry[1][3]))
            y5.append(float(entry[1][4]))
            #y6.append(float(entry[1][0]))
        break
'''

xnew = np.linspace(np.arange(len(x)).min(), np.arange(len(x)).max(), len(x))  # 300 represents number of points to make between T.min and T.max
#new_y1 = scipy.interpolate.UnivariateSpline(np.arange(len(x)), y1)(xnew)
new_y1 = scipy.interpolate.interp1d(np.arange(len(x)), y1)(xnew)
#new_y2 = scipy.interpolate.UnivariateSpline(np.arange(len(x)), y2)(xnew)
new_y2 = scipy.interpolate.interp1d(np.arange(len(x)), y2)(xnew)
#new_y3 = scipy.interpolate.UnivariateSpline(np.arange(len(x)), y3)(xnew)
new_y3 = scipy.interpolate.interp1d(np.arange(len(x)), y3)(xnew)
#new_y4 = scipy.interpolate.UnivariateSpline(np.arange(len(x)), y4)(xnew)
new_y4 = scipy.interpolate.interp1d(np.arange(len(x)), y4)(xnew)
#new_y5 = scipy.interpolate.UnivariateSpline(np.arange(len(x)), y5)(xnew)
new_y5 = scipy.interpolate.interp1d(np.arange(len(x)), y5)(xnew)
#new_y6 = scipy.interpolate.UnivariateSpline(np.arange(len(x)), y6)(xnew)
#new_y6 = scipy.interpolate.interp1d(np.arange(len(x)), y6)(xnew)



font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20,
        }
fig = pl.figure()
A1 = fig.add_subplot(111)
plot11, = A1.plot(np.arange(len(x)), y1, 'bs-', markeredgewidth=5,markeredgecolor='b', markersize=2, markerfacecolor='none', linewidth=1)# use pylab to plot x and y : Give your plots names
plot12, = A1.plot(np.arange(len(x)), y2, 'go--', markeredgewidth=5,markeredgecolor='g', markersize=2, markerfacecolor='none', linewidth=1)
plot13, = A1.plot(np.arange(len(x)), y3, 'y^-', markeredgewidth=5,markeredgecolor='y', markersize=2, markerfacecolor='none', linewidth=1)# use pylab to plot x and y : Give your plots names
plot14, = A1.plot(np.arange(len(x)), y4, 'r>-', markeredgewidth=5,markeredgecolor='r', markersize=2, markerfacecolor='none', linewidth=1)# use pylab to plot x and y : Give your plots names
plot15, = A1.plot(np.arange(len(x)), y5, 'c<--', markeredgewidth=5,markeredgecolor='c', markersize=2, markerfacecolor='none', linewidth=1)
#plot16, = A1.plot(xnew, new_y6, 'y^-', markeredgewidth=3,markeredgecolor='y', markersize=2, markerfacecolor='none', linewidth=1)# use pylab to plot x and y : Give your plots names

index_ls = []
scale_ls = []
real_x = []

number = 0
for i in x:
    real_x.append(number)
    if number % 4 == 0:
        index_ls.append(i)
        scale_ls.append(number)
    number += 1

print x
print index_ls
print scale_ls

s = 10
print x[s]
print real_x[s]
print [y1[s], y2[s], y3[s], y4[s], y5[s]]

s = 40
print x[s]
print real_x[s]
print [y1[s], y2[s], y3[s], y4[s], y5[s]]


s = np.array(y2).argmax()
print x[s]
print real_x[s]
print y2[s]

s = np.array(y2).argmin()
print x[s]
print real_x[s]
print y2[s]

arr = np.array(y1 + y2 + y3 + y4 + y5)

min = arr.min()*0.8
max = arr.max()*1.2

print max, min

xminorLocator = MultipleLocator(10)
yminorLocator = MultipleLocator((max-min)/10)
A1.xaxis.set_minor_locator(xminorLocator)
A1.yaxis.set_minor_locator(yminorLocator)
A1.xaxis.grid(True, which='minor',linestyle='--', linewidth=0.5)
A1.yaxis.grid(True, which='minor',linestyle='--', linewidth=0.5)

pl.title('', fontdict=font)# give plot a title

pl.xticks(fontsize=15)
pl.xticks(scale_ls, index_ls)
pl.xticks(rotation=90)
#xt = [round(i, 2) for i in np.arange(np.array(scale_ls).min(), np.array(scale_ls).max(), 10)]
#pl.xticks(xt)
pl.xlabel('Time', fontdict=font)# make axis labels
A1.xaxis.set_label_coords(1., -0.025)
pl.xlim(np.array(scale_ls).min()-1,np.array(scale_ls).max()+5)# set axis limits

pl.yticks(fontsize=15)
yt = [round(i, 2) for i in np.linspace(min, max, 5)]
pl.yticks(yt)
pl.ylabel('Probability', fontdict=font)
pl.ylim(min, max)

pl.legend([plot11, plot12, plot13, plot14, plot15], ('Topic 1','Topic 2', 'Topic 3', 'Topic 4', 'Topic 5'), ncol=1,  bbox_to_anchor=[0.9, .5, 0.1, 0.3], loc='upper right', fontsize=15)

pl.axvline(x=10, ymin=0.1, ymax = 0.91, linewidth=1.5, color='k')
pl.axvline(x=40, ymin=0.1, ymax = 0.83, linewidth=1.5, color='k')

A1.annotate('Probability = 0.51', xy=(11, 0.51), size=20, color='red', arrowprops=dict(facecolor='red', arrowstyle='->'), xytext=(16, 0.49))
A1.annotate('Probability = 0.11', xy=(76, 0.11), size=20, color='red', arrowprops=dict(facecolor='red', arrowstyle='->'), xytext=(60, 0.075))

pl.text(1, 0.57, 'Topic Distribution = [0.19, 0.18, 0.27, 0.15, 0.21]', size=20, style='italic', color='black')
pl.text(30, 0.53, 'Topic Distribution = [0.28, 0.18, 0.17, 0.13, 0.24]', size=20, style='italic', color='black')



pl.show()