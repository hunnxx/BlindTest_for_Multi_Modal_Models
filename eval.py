import re
import json

from typing import List

class eval_metric:
    @staticmethod
    def func_subway_connections(data):
        correct_cnt = 0
        total_cnt = len(data)
        word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }

        for item in data:
            gt = item['gt']
            ans = item['answer'].lower()

            pattern = r'\{(\d+)\}|\b(\d+)\b|(\b(zero|one|two|three|four|five|six|seven|eight|nine)\b)'

            match = re.search(pattern, ans)

            revised_ans = None
            if match:
                if match.group(1):
                    revised_ans = match.group(1)
                elif match.group(2):
                    revised_ans = match.group(2)
                elif match.group(3):
                    revised_ans = word_to_num.get(match.group(3))

            if gt == revised_ans:
                correct_cnt += 1

        return correct_cnt, total_cnt

    @staticmethod
    def func_nested_squares(data):
        correct_cnt = 0
        total_cnt = len(data)
        word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }

        for item in data:
            gt = item['gt']
            ans = item['answer'].lower()

            pattern = r'\{(\d+)\}|\b(\d+)\b|(\b(zero|one|two|three|four|five|six|seven|eight|nine)\b)'

            match = re.search(pattern, ans)

            revised_ans = None
            if match:
                if match.group(1):
                    revised_ans = match.group(1)
                elif match.group(2):
                    revised_ans = match.group(2)
                elif match.group(3):
                    revised_ans = word_to_num.get(match.group(3))

            if gt == revised_ans:
                correct_cnt += 1

        return correct_cnt, total_cnt

    @staticmethod
    def func_line_plot_intersections(data):
        correct_cnt = 0
        total_cnt = len(data)
        word_to_num = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'once': '1', 'twice': '2', 'no': '0', 'not': '0'
        }

        for item in data:
            gt = item['gt']
            ans = item['answer'].lower()

            pattern = r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|once|twice|no|not)\b'
            match = re.search(pattern, ans)
            revised_ans = None
            if match:
                revised_ans = word_to_num.get(match.group(1)) if match.group(1).isalpha() else match.group(1)

            if gt == revised_ans:
                correct_cnt += 1

        return correct_cnt, total_cnt

    @staticmethod
    def func_touching_circles(data):
        correct_cnt = 0
        total_cnt = len(data)

        for item in data:
            gt = item['gt']
            ans = item['answer'].lower()

            pattern = r'\b(yes|are touching|no|not)\b'
            match = re.search(pattern, ans)
            revised_ans = None
            if match:
                revised_ans = match.group(1)
                if revised_ans == 'are touching':
                    revised_ans = 'yes'
                elif revised_ans == 'not':
                    revised_ans = 'no'
                revised_ans = revised_ans.capitalize()

            if gt == revised_ans:
                correct_cnt += 1
        
        return correct_cnt, total_cnt

    @staticmethod
    def func_counting_grid(data):
        correct_cnt = 0
        total_cnt = len(data)
        word_to_num = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
        }

        for item in data:
            gt = item['gt']
            ans = item['answer'].lower()

            gt_row, gt_col = gt.split(',')

            pattern = r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b'
            matches = re.findall(pattern, ans)
            revised_ans = [word_to_num.get(match, None) if match.isalpha() else match for match in matches]
            if len(revised_ans) == 2 and (gt_row == revised_ans[0]) and (gt_col == revised_ans[1]):
                correct_cnt += 1

        return correct_cnt, total_cnt

    @staticmethod
    def func_olympic_counting(data):
        word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }

        correct_cnt = 0
        total_cnt = len(data)

        for item in data:
            gt = item['gt'].lower()
            ans = item['answer'].lower()

            pattern = r'\{(\d+)\}|\b(\d+)\b|(\b(zero|one|two|three|four|five|six|seven|eight|nine)\b)'
            match = re.search(pattern, ans)

            revised_ans = None
            if match:
                if match.group(1):
                    revised_ans = match.group(1)
                elif match.group(2):
                    revised_ans = match.group(2)
                elif match.group(3):
                    revised_ans = word_to_num.get(match.group(3))

            if gt == revised_ans:
                correct_cnt += 1

        return correct_cnt, total_cnt

    @staticmethod
    def func_circled_letter(data: List):
        correct_cnt = 0
        total_cnt = len(data)

        for item in data:
            gt = item['gt'].lower()
            ans = item['answer'].lower()

            pattern = r'[\*\(\{\'\"})]+\s?([a-zA-Z])\s?\.?[\*\)\}\'\"})]+|\b\s([a-zA-Z])\.|\b\s[a-zA-Z]\s\b'
            match = re.search(pattern, ans)

            revised_ans = None
            if match:
                revised_ans = [group for group in match.group() if group.isalpha()][0]
            
            if gt == revised_ans:
                correct_cnt += 1
        
        return correct_cnt, total_cnt

def load_json_file(path: str = None):
    if path is None:
        raise Exception('Path is None')
    
    with open(path, 'r') as file:
        json_data = json.load(file)
    return json_data

def main():
    for path in ['data_7k/blindtest_chameleon-7b_bfloat16.json', 'data_7k/blindtest_chameleon-7b_float32.json',
                 'data_7k/blindtest_idefics-9b_bfloat16.json', 'data_7k/blindtest_idefics-9b_float32.json',
                 'data_7k/blindtest_idefics-9b-instruct_bfloat16_mod.json', 'data_7k/blindtest_idefics-9b-instruct_float32_mod.json',
                 'data_7k/blindtest_llava-1.5-7b-hf_bfloat16.json', 'data_7k/blindtest_llava-1.5-7b-hf_float32.json',
                 'data_7k/blindtest_kosmos-2-patch14-224_bfloat16.json', 'data_7k/blindtest_kosmos-2-patch14-224_float32.json',]:
        print("="*20 + path + "="*20)
        json_data = load_json_file(path)
        for task_name, task_data in json_data.items():
            revised_task_name = task_name.lower().split('-')[0].rstrip(' ').replace(' ', '_')
            eval_func = getattr(eval_metric, f'func_{revised_task_name}')
            correct_cnt, total_cnt = eval_func(task_data)

            print(f'{task_name}\n{correct_cnt}/{total_cnt} ({(correct_cnt/total_cnt)*100:.2f}%)')

# def modify(path = 'data_7k/blindtest_idefics-9b-instruct_float32.json'):
#     new_json_data = {}
#     json_data = load_json_file(path)
#     for task_name, task_data in json_data.items():
#         if not task_name in new_json_data.keys():
#             new_json_data[task_name] = []

#         for item in task_data:
#             output = item['output']
#             item['answer'] = output.split('Assistant:')[1]
#             new_json_data[task_name].append(item)
        
#     with open('data_7k/blindtest_idefics-9b-instruct_float32_mod.json', 'w') as file:
#         json.dump(new_json_data, file)
            

if __name__ == '__main__':
    main()