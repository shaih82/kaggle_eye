from Utils.data_layer_read import *
from Utils import caffenet
from Utils import save_net, load_net, pp_epoch, show_im
from Utils import HOME
from sklearn import cross_validation
import scipy

import pdb

print "tst"

eyes = np.loadtxt(HOME+'/kaggle_eye/sampleLabels.csv',delimiter=',', skiprows=1,
                       # usecols=range(5)
                       dtype={'names':['im','y'],
                              'formats':['|S15','<i1']}
                       )

x = eyes['im']
x = map(lambda x: HOME+'/kaggle_eye/sample/%s.jpeg'%(x,), x)
y = eyes['y']

x, x_test, y, y_test = cross_validation.train_test_split(x,y, test_size=0.2, random_state=0)

lines_train = zip(x,y)
lines_test = zip(x_test,y_test)

mean_im = np.zeros((3,1,1))

print "tst2"

# mean_im = calc_mean(get_lines(train_aligned_path)[:20000])
# np.save(HOME+'/WebFace/mean_align_227_img.npy', mean_im)

#load data
# mean_im = np.load(HOME+'/WebFace/mean_align_227_img.npy')
# #TODO: check if mean RGB value works better
# mean_im = mean_im.mean((1,2)).reshape(3,1,1)
batch_size = 1*2
(c , w, h ) = (3,600,600)



def rgb2grey(im):
    rW=0.2126
    gW=0.7152
    bW = 1. - rW -gW
    return (rW*im[:,:,0] + gW*im[:,:,1] + bW*im[:,:,2])

def preImage(im):
    return scipy.misc.imresize(  rgb2grey(im.transpose(1,2,0)) , (w,h))

print "tst3"

it_train = RandomBatchIterator(lines_train,
                               mean_im,
                               batch_size=batch_size,
                               # post=lambda x:x[:,50:250, 50:250]
                               post=preImage
                               )
it_test = RandomBatchIterator(lines_test,
                              mean_im,
                              batch_size=batch_size,
                              # post=lambda x:x[:,50:250, 50:250]
                              post=preImage
                              )
pdb.set_trace()


it_test.next_batch()
it_train.next_batch()


#create network
input_shape = (batch_size, c, w , h  ) 
#it_train.x.get_value().shape
output_dim = 5
net = caffenet.get_BN_net(input_shape, output_dim)
# import cPickle as pickle
# with open(HOME+'/webface_front_BN_22112_iter_14000.pickle','r') as f:
#     net = pickle.load(f)

#compile
func = create_iter_functions(it_train,it_test,
                             output_layer=net, 
                             solver=lasagne.updates.nesterov_momentum,
                             learning_rate=0.05)
debug = debug_input(net,it_train)
num_epochs = 20*1000

#train
import pylab
for i,epoch in train_loop(func,it_train,it_test, net):
    pp_epoch(epoch)

    # im = debug()[0][0]
    # show_im(np.asarray(im)+mean_im)
    # pylab.draw()
    if i%1000 == 0:
        import cPickle as pickle
        with open(HOME+'/webface_align_227_iter_'+str(i)+'.pickle','wb') as f:
            pickle.dump(net,f,protocol=pickle.HIGHEST_PROTOCOL)
        # save_net(net,'webface_align_BN_iter_'+str(i)+'.pickle')
          
    if i >= num_epochs:
        break

bp=3

