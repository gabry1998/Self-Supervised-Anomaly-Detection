from email.mime.text import MIMEText
from self_supervised.functional import get_all_subject_experiments
from training import run
from evaluator import evaluate
from datetime import datetime
import smtplib 
import numpy as np



def get_textures_names():
    return ['carpet','grid','leather','tile','wood']

def get_obj_names():
    return [
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut',
        'pill',
        'screw',
        'tile',
        'toothbrush',
        'transistor',
        'zipper'
    ]

def obj_set_one():
    return [
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut']

def obj_set_two():
    return [
        'pill',
        'screw',
        'toothbrush',
        'transistor',
        'zipper']

def specials():
    return [
        'cable',
        'capsule',
        'pill',
        'screw']


if __name__ == "__main__":
    inputdir = 'brutta_brutta_copia/computations/'
    outputdir = 'brutta_brutta_copia/computations/'
    experiments = get_all_subject_experiments('dataset/')
    textures = get_textures_names()
    obj1 = obj_set_one()
    obj2 = obj_set_two()
    
    #### modificare qui ####
    experiments_list = obj1
    #### -------------- ####
    
    subjects = np.array_str(np.array(experiments_list))[0:-1].replace(' ','<br>- ').replace('[','- ')
    # start training
    now = datetime.now()
    start = now.strftime("%d/%m/%Y %H:%M:%S")
    run(
        experiments_list=experiments_list,
        dataset_dir='dataset/', 
        root_outputs_dir=inputdir,
        imsize=(256,256),
        patch_localization=True,
        batch_size=64,
        projection_training_lr=0.03,
        projection_training_epochs=10,
        fine_tune_lr=0.01,
        fine_tune_epochs=50
    )
    
    # end training, notify
    
    now = datetime.now()
    end = now.strftime("%d/%m/%Y %H:%M:%S")
    msg = '''\
    <html>
    <head>
    Training completato. <br>
    Inizio: {start} <br>
    Fine: {end} <br>
    Training effettuato su: <br> 
    {objs}
    <br>
    (orario e' un'ora indietro) <br>
    </head>
    <body>
    </body>
    </html>'''.format(start=start, end=end, objs=subjects)
    text = MIMEText(msg,'html')
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:  
        email_address = 'server.lab.peranet@gmail.com'
        email_password = 'qzpykdqppygcfqtm'
        connection.login(email_address, email_password)
        connection.sendmail(
            from_addr=email_address, 
            to_addrs='gabrymad998@gmail.com', 
            msg='subject:Training \n'+text.as_string())
        
    # start evaluation
    now = datetime.now()
    start = now.strftime("%d/%m/%Y %H:%M:%S")
    report = evaluate(
        dataset_dir='dataset/',
        root_inputs_dir=inputdir,
        root_outputs_dir=outputdir,
        imsize=(256,256),
        patch_dim = 32,
        stride=8,
        seed=123456789,
        patch_localization=True,
        experiments_list=experiments_list
    )
    # end evaluation, notify
    now = datetime.now()
    end = now.strftime("%d/%m/%Y %H:%M:%S")
    msg = '''\
    <html>
    <head>
    Test completati. <br>
    Inizio: {start} <br>
    Fine: {end} <br>
    Test effettuati su: <br> 
    {objs}
    <br>
    (orario e' un'ora indietro) <br>
    </head>
    <body>
    {df} 
    </body>
    </html>'''.format(df=report.to_html(), start=start, end=end, objs=subjects)
    text = MIMEText(msg,'html')
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:  
        email_address = 'server.lab.peranet@gmail.com'
        email_password = 'qzpykdqppygcfqtm'
        connection.login(email_address, email_password)
        connection.sendmail(
            from_addr=email_address, 
            to_addrs='gabrymad998@gmail.com', 
            msg="subject:Risultati \n"+text.as_string())