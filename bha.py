from phba import Imagine

"""
num_imgs = 3
num_prompts = 7
eps = num_imgs*num_prompts
img_vector = [0] * eps
#prompts = 'confusion. joy. melancholy'
prompts = 'profound beauty. anguish-greif. painful confusion. joyous release. calm contemplation. deep longing. ecstatic melancholy'

for x in range(eps):
    #if x < 10:
    #    img_vector[x] ="i/0" + str(x%7) + ".jpeg"
    #else:
    #    img_vector[x] = "i/" + str(x%7) + ".jpeg"
    #img_vector[x] ="i/0" + str(x%7) + ".jpeg"
    img_vector[x] ="i/" + str(x%num_imgs) + ".jpeg"
    
for z in range(num_imgs-1):
#    prompts ='confusion. joy. melancholy.' + prompts
     prompts ='profound beauty. anguish-grief. painful confusion. joyous release. calm contemplation. deep longing. ecstatic melancholy.' + prompts
            
"""

icount = 3
img_vector = [0] * icount
for x in range(icount):
	img_vector[x] = "i/" + str(x) + ".jpeg"

tcount = 7	
txt_vector = ["magical-real-power", "fantasy as reality", "empathetic-awareness", "paiful-delight", "internal self-romance", "touching with song", "the universal-language of movement"]
#tcount = 1
txt_vector = ["a modern minimal painting of a frog-woman in a black dress and a pearl necklace "]


imagine = Imagine( 
    
    text = "magic",
    #bg_txt='a lesbian floral-elf-woman on a journey through magical aurora-chambers and pools of shadow-mystery, she sings poems along the path to light the way',

    image_width=540,
    image_height=540,
    
    #image_width=640,
    #image_height=360,
    
    #image_width=426,
    #image_height=240,
    
    num_layers=26,
    batch_size=27,
    #pre new clip model
    #num_layers=34,
    #batch_size=33,
    #these work with 512
    #num_layers=28,
    #batch_size=29,
    
    seed=1717913,
    theta_initial=23,
    theta_hidden=30,
    
    t_num = 1,
    i_num = 3,
    i_freq=21,
    t_freq=76,
    iterations=500,
    epochs=26,
    gradient_accumulate_every=1,
    
    do_vqcuts=False,
    
    change_lr=True,
    #change_lr=False,
    lr=10e-06,
    lr_max=13e-06,
    
    #averaging_weight=0.03,
    averaging_weight=0.3,   #original
    
    #use_flow=False,
    use_flow=True,
    #use_flow_fx = True,
    use_flow_fx = False,
    
    #bg_img = 'i/bg.jpeg',
    flow_txts = txt_vector,
    #flow_imgs = img_vector,
    
    use_flow_txt_offset = False,
    flow_img_wt = 0.6,
    flow_img_wt_range = 0,
    bg_img_wt = 0.666,
    bg_img_wt_range = 0.333,
    bg_wt=0.5,
    bg_wt_range = 0.1,
    
    #seemless = True,
    seemless = False,
    ifade_range =  13,
    tfade_range = 7,  
    
    create_story = False,
	story_separator = '.',
    
    #img="img/ab_firepeople.jpeg",
    #img2="imgs2use/ab_bw.jpeg",
    
    do_cutout = True,
    #center_bias=True,
    lower_bound_cutout=0.1, # should be smaller than 0.8 #=0.1
    saturate_bound=False,
    #saturate_limit=0.5,
    
    save_every=1,
    save_progress=True,
    open_folder=False,
    
    #gauss_sampling=True,
    #seed=1729,
    #start_image_path="imgs2use/ab_angel.jpeg",
    #start_image_train_iters=100,
    #start_image_lr=0.2e-4,
    #model_name="ViT-B/16",
)

imagine()
