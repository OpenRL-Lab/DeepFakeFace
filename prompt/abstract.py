import pickle
import os


class Abstract():
    def __init__(self, prompt_pth, data_pth=None, store=True,num=100,each=1) -> None:
        self.prompt_pth = prompt_pth
        self.num = num
        self.each = each
        if os.path.exists(prompt_pth):
            self.prompt = self.load(prompt_pth)
        else:
            if not data_pth:
                assert 'datapth should not be none'
            self.prompt = self.generate(data_pth)
            if store:
                self.save(self.prompt,prompt_pth)
        

    def generate(self, data_pth):
        return None

    def save(self, prompt,path):
        with open(path, 'wb') as f:
            pickle.dump(prompt, f)

    def load(self, path):
        with open(path, 'rb') as f:
            result = pickle.load(f)
        return result
    
    def alter(self,prompts):
        dic = {}
        result = []
        for prompt in prompts:
            new_prompt = prompt.replace(', ', '_')
            if new_prompt not in dic:
                dic.update({new_prompt: 0})
            else:
                dic[new_prompt] += 1
            result.append(new_prompt+'_'+str(dic[new_prompt])+'.jpg')
        return result
