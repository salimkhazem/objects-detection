import os 

Root = "/home/salimkhazem/workspace/capgemini/Masks_nuts/"
for f in sorted(os.listdir(Root)): 
    os.rename(r'/home/salimkhazem/workspace/capgemini/Masks_nuts/'+ str(f), r'/home/salimkhazem/workspace/capgemini/Masks_nuts/' + str(f[:6]) + '_mask.png')
    
