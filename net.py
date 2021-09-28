import torch
from torch import nn
import torch.nn.functional as F
import vgg
from newnl import ImprovedNonlocal


def convblock(in_,out_,ks,st,pad):
    return nn.Sequential(
        nn.Conv2d(in_,out_,ks,st,pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )
    
class Decoder(nn.Module):
    def __init__(self,in_1,in_2):
        super(Decoder, self).__init__()
        self.nl1 = ImprovedNonlocal(in_1,in_1,64,64,dropout=0.05, sizes=([1]), norm_type='batchnorm')
        self.conv1 = convblock(in_1,128, 3, 1, 1)
        self.conv_global = convblock(512,128,3, 1, 1)
        self.nl2 = ImprovedNonlocal(in_2,in_2,64,64,dropout=0.05, sizes=([1]), norm_type='batchnorm')
        self.conv_curfeat =convblock(in_2,128,3,1,1)
        self.conv_out= convblock(128,in_2,3,1,1)

    def forward(self, pre,cur,global_info):
        cur_size = cur.size()[2:]
        pre = self.nl1(pre)
        pre =self.conv1(F.interpolate(pre,cur_size,mode='bilinear',align_corners=True))

        global_info = self.conv_global(F.interpolate(global_info,cur_size,mode='bilinear',align_corners=True))
        cur_feat =self.conv_curfeat(self.nl2(cur))
        fus = pre + cur_feat + global_info
        return self.conv_out(fus)

        
class GlobalAttention(nn.Module):
    def __init__(self):
        super(GlobalAttention, self).__init__()

        self.rgb_c  = convblock(512,512,3,1,1)
        self.t_c  = convblock(512,512,3,1,1)
        self.de_channel = nn.Sequential(
            nn.AdaptiveMaxPool2d(25),
            nn.Conv2d(1024,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.b0 = nn.Sequential(
            nn.AdaptiveMaxPool2d(13),
            nn.Conv2d(256,128,1,1,0,bias=False),
            nn.ReLU(inplace=True)
        )

        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(9),
            nn.Conv2d(256,128,1,1,0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(5),
            nn.Conv2d(256, 128, 1, 1, 0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 128, 1, 1, 0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.fus = convblock(768,512,1,1,0)
        
    def forward(self, rgb1,rgb2,t1,t2):
        x_size=rgb1.size()[2:]
        rgb1_50 = F.interpolate(self.rgb_c(rgb1),[50,50],mode='bilinear',align_corners=True)
        rgb=torch.mul(rgb1_50,rgb2)
        t1_50 = F.interpolate(self.t_c(t1),[50,50],mode='bilinear',align_corners=True)
        t=torch.mul(t1_50,t2)
        x=self.de_channel(torch.cat((rgb,t),1))
        x0 = F.interpolate(self.b0(x),x_size,mode='bilinear',align_corners=True)
        x1 = F.interpolate(self.b1(x),x_size,mode='bilinear',align_corners=True)
        x2 = F.interpolate(self.b2(x),x_size,mode='bilinear',align_corners=True)
        x3 = F.interpolate(self.b3(x),x_size,mode='bilinear',align_corners=True)
        out = self.fus(torch.cat((x0,x1,x2,x3,x),1))
        return rgb,t,out
        

class SaliencyNet(nn.Module):
    def __init__(self):
        super(SaliencyNet, self).__init__()

        self.global_info = GlobalAttention()
        self.sal_global = nn.Conv2d(512, 1, 1, 1, 0)

        self.d4_r = Decoder(512,512)
        self.d3_r= Decoder(512,256)
        self.d2_r= Decoder(256,128)

        self.d4_t = Decoder(512, 512)
        self.d3_t = Decoder(512, 256)
        self.d2_t = Decoder(256, 128)

        self.sal_r=nn.Conv2d(128, 1, 1, 1, 0)
        self.sal_t = nn.Conv2d(128, 1, 1, 1, 0)


    def forward(self,rgb,t):
        xsize=rgb[0].size()[2:]
        rgb_f,t_f,global_info = self.global_info(rgb[4],rgb[3],t[4],t[3])
        d1=self.d4_r(rgb_f,rgb[3],global_info)
        d2=self.d4_t(t_f, t[3], global_info)
        d3 = self.d3_r(d1,rgb[2],global_info)
        d4 = self.d3_t(d2, t[2], global_info)
        d5 = self.d2_r(d3, rgb[1], global_info)
        d6 = self.d2_t(d4, t[1], global_info) 

        result_g = self.sal_global(global_info)

        result_r=self.sal_r(F.interpolate(d5,xsize,mode='bilinear',align_corners=True))
        result_t = self.sal_t(F.interpolate(d6, xsize, mode='bilinear', align_corners=True))

        return result_r,result_t,result_g

class RGBTSODnet(nn.Module):
    def __init__(self):
        super(RGBTSODnet,self).__init__()
        self.rgb_net= vgg.a_vgg16()
        self.t_net= vgg.a_vgg16()  
        self.s_net=SaliencyNet()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,rgb,t):
        # rgb = input[:,:3]
        # t = input[:,3:]
        rgb_f= self.rgb_net(rgb)
        t_f= self.t_net(t)
        result_r,result_t,result_g = self.s_net(rgb_f,t_f)
        return result_r,result_t,result_g

    def load_pretrained_model(self):
        st=torch.load("model/vgg16.pth")
        st2={}
        for key in st.keys():
            st2['base.'+key]=st[key]
        self.rgb_net.load_state_dict(st2)
        self.t_net.load_state_dict(st2)
        print('loading pretrained model success!')


if __name__=="__main__":
    a=torch.rand(1,3,400,400)
    b=a
    net=RGBTSODnet()
    c,c1,c2=net(a,b)
    print(c.shape)
    print(c1.shape)
    print(c2.shape)