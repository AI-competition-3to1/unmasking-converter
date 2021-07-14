class dataset_for_unet(data.Dataset):
  def __init__(self,data_path):
    self.images = data_path +'_masked'
    self.masks = data_path + '_mask'
    self.fns=[]
    image_list=os.listdir(self.images)
    mask_list=os.listdir(self.masks)
    for i,_ in enumerate(image_list):
      self.fns.append([(self.images+'/'+image_list[i]),(self.masks+'/'+mask_list[i])])
  def __getitem__(self,index):
    image_path,mask_path=self.fns[index]
    i=cv2.imread(image_path,cv2.IMREAD_COLOR)
    i=cv2.resize(i,dsize=(256, 256)).astype(np.float32)/255
    m= cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    m=cv2.resize(m,dsize=(256, 256))
    np.around(m)
    m = np.expand_dims(m, axis=0)
    i = torch.from_numpy(i).permute(2,0,1).contiguous()
    m = torch.from_numpy(m).contiguous()
    return i, m
  def collate_fn(self, batch):
        i = torch.stack([i[0] for i in batch])
        m = torch.stack([i[1] for i in batch])
        return {
            'imgs': i,
            'masks':  m
        }
    
  def __len__(self):
      return len(self.fns)