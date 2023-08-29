from .abstract import Abstract
from scipy.io import loadmat
import random


class WIKI(Abstract):
    def __init__(self, prompt_pth, data_pth=None, store=True, num=100, each=1) -> None:
        super().__init__(prompt_pth, data_pth, store, num, each)
        random.seed(2022)

    def generate(self, data_pth):
        paths = []
        prompts = []
        pairs = []
        path2prompt = {}
        male = []
        female = []
        data = loadmat(data_pth)['wiki'][0][0]
        for capture_year, file_name, gender, celeb_name in zip(data[1][0], data[2][0], data[3][0], data[4][0]):
            try:
                birth_year = file_name[0].split(
                    '.')[0].split('_')[1].split('-')[-3]
                age = int(capture_year)-int(birth_year)
                if age < 0:
                    continue
                prompt = celeb_name[0]+', ' + \
                    'celebrity, ' + str(age)+'-year-old'
                path2prompt.update({file_name[0]: prompt})
                if gender == 1:
                    male.append(file_name[0])
                elif gender == 0:
                    female.append(file_name[0])
            except:
                print('there is error, capture year is: ', capture_year,
                      ' file name is: ', file_name, ' celeb name is: ', celeb_name)
        random.shuffle(male)
        random.shuffle(female)
        for idx in range(len(male)//2):
            paths.append(male[idx*2])
            prompts.append(path2prompt[male[idx*2+1]])
            paths.append(male[idx*2+1])
            prompts.append(path2prompt[male[idx*2]])
            pairs.extend([(male[idx*2], male[idx*2+1]),
                         (male[idx*2+1], male[idx*2])])
        for idx in range(len(female)//2):
            paths.append(female[idx*2])
            prompts.append(path2prompt[female[idx*2+1]])
            paths.append(female[idx*2+1])
            prompts.append(path2prompt[female[idx*2]])
            pairs.extend([(female[idx*2], female[idx*2+1]),
                         (female[idx*2+1], female[idx*2])])
        return (paths, prompts, pairs)
