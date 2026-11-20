import json
import random
import string
import re
from hymm.utils.distributed_utils import rank0_print

# parse json to caption for QWEN
class CaptionAug:
    """
    对caption进行数据增广,包括:
    1. 配置对IP数据替换的比例, 默认配置为全部替换(长短caption)
    2. 配置比例删除文字，去除可能存在错误的文字(长caption)
    3. 对结构化caption进行随机组合。背景, 镜头, 风格, 景别, 氛围, 运镜(视频)+[长文本]/[短文本], 字幕, 高质, 影视类别

    """

    def __init__(self,
                 caption_sample_ratio=None,
                 ip_remove_rate=1.0,
                 text_remove_rate=1.0,
                 long_short_caption_mutex=True,
                 ip_tag_mapping_file=None,
                 ocr_only_long_caption=False,
                 logger=None,
                 ):
        """
        Args:
            caption_sample_ratio : 结构化caption采样比例
            ip_remove_rate (float, optional): IP数据替换的比例
            text_remove_rate (float, optional): 文字替换的比例
            long_short_caption_mutex : 长短caption互斥
            ip_tag_mapping_file: IP字段映射名称
            ocr_only_long_caption: 当long caption中包含OCR时，只使用long caption
        """
        if logger is None:
            from loguru import logger
        self.logger = logger

        self.ip_tag_map = json.loads(open(ip_tag_mapping_file, "r").read()) if ip_tag_mapping_file else {}
        if len(self.ip_tag_map) > 0:
            self.use_ip_tag_map = True
        else:
            self.use_ip_tag_map = False
        self.ip_remove_rate = ip_remove_rate
        self.text_remove_rate = text_remove_rate
        self.caption_sample_ratio = self.normalize_dict_keys(caption_sample_ratio)
        self.caption_sample_ratio_keys = {
            "long caption", "medium caption", "short caption", "background", "shot type", "style",
            "light", "atmosphere", "camera movement", "with subtitle", "without subtitle", "high quality", "movie category",
            'medium caption', 'composition', 'IP', # for image caption v2
        }
        self.long_short_caption_mutex = long_short_caption_mutex
        self.logger = logger
        self.ocr_only_long_caption = ocr_only_long_caption

        # default
        # self.caption_sample_ratio = {key: 0.8 for key in self.caption_sample_ratio_keys}, w/wo subtitle 除外
        # self.caption_sample_ratio["long caption"] = 0.5
        # self.caption_sample_ratio["short caption"] = 0.5

        # 关于字幕的两个采样比例，单独设置
        if 'with subtitle' not in self.caption_sample_ratio:
            self.caption_sample_ratio['with subtitle'] = 1.0
        if 'without subtitle' not in self.caption_sample_ratio:
            self.caption_sample_ratio['without subtitle'] = 0.5

        for key in self.caption_sample_ratio:
            if key not in self.caption_sample_ratio_keys:
                raise NotImplementedError(
                    f"{key} not in caption_sample_ratio_keys, all sample ratio keys: {self.caption_sample_ratio_keys}.")

        rank0_print(self.logger,
            "CaptionAug using caption sample ratio: {} ip_remove_rate:{} text_remove_rate:{} ocr_only_long_caption:{}".format(
                json.dumps(self.caption_sample_ratio, indent=4), ip_remove_rate, text_remove_rate,
                ocr_only_long_caption))

    def safe_load_string(self, text):
        """加载结构化caption数据, json or xml格式"""
        try:
            text = json.loads(text)
        except json.decoder.JSONDecodeError:
            text = eval(text)
        # except Exception as e:
        #     raise NotImplementedError("json str format only {}, got str: {}".format(str(e), text))

        if type(text) is str:
            try:
                text = eval(text)
                if type(text) is str:
                    text = text.replace("'", "\"")
                    text = json.loads(text)
            except Exception as e:
                raise NotImplementedError("json str format only {}, got eval str: {}".format(str(e), text))

        if type(text) != dict:
            print('load caption error', text)

        return text

    def random_drop_ip(self, long_short_caption):
        """是否随机丢弃对IP的描述"""
        if self.ip_remove_rate == 0 or random.random() > self.ip_remove_rate:
            return long_short_caption

        for key in self.ip_tag_map:
            if key in long_short_caption:
                long_short_caption = long_short_caption.replace(key, self.ip_tag_map[key])
        return long_short_caption

    def random_drop_word(self, long_caption):
        # self.text_remove_rate = 0 # TODO: to delete, only for ocr train
        # print(f"0000000000 text_remove_rate == {self.text_remove_rate}")
        """是否随机丢弃文字部分描述"""
        if self.text_remove_rate == 0 or random.random() > self.text_remove_rate:
            return long_caption

        """有一些句号在引号中间，在引号外面的句号才split"""

        def split_by_unquoted_periods(text):
            result = []
            current_segment = ""
            in_quotes = False

            for char in text:
                if char == '"':
                    in_quotes = not in_quotes

                if char == '.' and not in_quotes:
                    result.append(current_segment)
                    current_segment = ""
                else:
                    current_segment += char

            if current_segment:
                result.append(current_segment)

            return result

        def if_caption_has_ocr(text: str):
            try:
                import re
                _ocr_in_caption = re.findall(r' "([^"]*)"', text)
                if len(_ocr_in_caption) > 0:
                    return True
            except Exception as e:
                # print(e)
                # pdb.post_mortem()
                return False
            return False

        def find_ocr_and_drop(text):
            items = split_by_unquoted_periods(text)
            if len(items) == 1:
                return text

            valid_items = []
            for item in items:
                if "\"" in item and any([_x in item.lower() for _x in
                                         ["reads", "reading", "written", "title", "writes", "text", "words",
                                          "chinese characters"]]):  # 包含引号并且包含某些特殊字符串
                    continue
                if if_caption_has_ocr(item):
                    continue
                valid_items.append(item)
            if len(valid_items) == 0:
                return text
            result = ".".join(valid_items)
            if result and result[-1] not in string.punctuation:
                result += '.'
            return result

        return find_ocr_and_drop(long_caption)

    def get_source_tag_mapping(self):
        source_rename_mapping = {
            "caimai_jilupian": "CMJLP",
            "1400T_video": "1400T",
            "1400T_video_anping_cos": "1400TVAC",
            "artlist_video": "ARTLIST",
            "699pic_video": "699PIC",
            "4kvideo": "4KVID",
            "highquality93w": "HQ93",
            "xpc": "XPC",
            "nature": "XPC",
            "artificial_object": "XPC",
            "artificial_scene": "XPC",
            "people": "XPC",
            "plant": "XPC",
            "transport": "XPC",
            "render": "XPC",
            "animal": "XPC",
            "logo": "XPC",
            "storyblocks_video": "STBLK",
            "pixabay_video": "PIXABAY",
            "videvo_video": "VIDEVO",
        }

        source_tag_mapping = {
            "caimai_jilupian": "Documentary",
            "1400T_video": "Cinematic",
            "1400T_video_anping_cos": "Cinematic",
            "artlist_video": "High-quality, HDR, High detailed",
            "699pic_video": "High-quality",
            "4kvideo": "4k, High-quality",
            "highquality93w": "High-quality",
            "xpc": "High-quality",
            "nature": "High-quality",
            "artificial_object": "High-quality",
            "artificial_scene": "High-quality",
            "people": "High-quality",
            "plant": "High-quality",
            "transport": "High-quality",
            "render": "High-quality",
            "animal": "High-quality",
            "logo": "High-quality",
            "storyblocks_video": "High-quality",
            "pixabay_video": "High-quality",
            "videvo_video": "High-quality",
            "transition": "Scene changes",
            "no_transition": "Scene does not change",
        }

        return source_rename_mapping, source_tag_mapping

    def get_hq_text(self, source, stage=None):
        """
        high quality tag
        """
        # 仅对stage 540p, 720p生效
        if stage not in ["360p", "540p", "720p"]:
            return None
        # special case for "highquality93w"
        if source.startswith("highquality93w"):
            source = "highquality93w"

        _, source_tag_mapping = self.get_source_tag_mapping()
        for source_name, hq_tag in source_tag_mapping.items():
            if source_name == source:
                tags = hq_tag.split(", ")
                random.shuffle(tags)
                hq_tag = ", ".join(tags)
                return hq_tag
        return None

    def get_hq_source_tag(self, source):
        # special case for "highquality93w"
        if source.startswith("highquality93w"):
            source = "highquality93w"

        source_rename_mapping, _ = self.get_source_tag_mapping()
        all_hq_sources = set(source_rename_mapping.keys())
        if source in all_hq_sources:
            return f"Video Source: [{source_rename_mapping[source]}]"
        return None

    def ocr_in_caption(self, caption):
        if re.findall(r' "([^"]*)"', caption):
            return True
        return False
    
    def random_compose(self, caption_dict):
        """随机组合caption"""
        tag_list = []
        long_short_caption = []
        all_captions = []
        hq_source_tag = None
        for key in caption_dict:
            """
            key的几种类别：
            1. caption类
            2. stage类: 忽略
            3. source类 (与stage同时使用)
            4. subtitle类
            5. 其他
            """
            if 'caption' in key:
                all_captions.append(caption_dict[key])
                if random.random() < self.caption_sample_ratio[key]:
                    long_short_caption.append(caption_dict[key])
            elif key == 'stage':
                # 不在这里处理，只在source处处理
                continue
            elif key == "source":
                hq_sample_ratio = self.caption_sample_ratio.get("high quality", 0.8)
                stage = caption_dict.get("stage")
                # 仅当stage满足需求时才使用
                hq_text = self.get_hq_text(caption_dict[key], stage)
                if hq_text and random.random() < hq_sample_ratio:
                    tag_list.append(hq_text)
                if hq_text and random.random() < hq_sample_ratio:  # 独立概率
                    hq_source_tag = self.get_hq_source_tag(caption_dict[key])
            elif 'subtitle' in key:
                """对于subtitle字段，需要单独处理，因为其中的两种结果需要的概率不同, 此时的key='has subtitle'"""
                has_subtitle = caption_dict[key]
                without_subtitle = 'without' in has_subtitle.lower()  # 总共有两种结果 With subtitle / Without subtitle
                with_subtitle = not without_subtitle
                if with_subtitle and random.random() < self.caption_sample_ratio['with subtitle']:
                    tag_list.append(has_subtitle)
                if without_subtitle and random.random() < self.caption_sample_ratio['without subtitle']:
                    tag_list.append(has_subtitle)
            else:
                if key in self.caption_sample_ratio and random.random() < self.caption_sample_ratio[key]:
                    tag = caption_dict[key].strip()
                    if tag == "" or tag.lower() == "none":
                        continue
                    tag_list.append(tag)

        has_long_caption = 'long caption' in caption_dict
        long_caption = caption_dict.get('long caption')
        if self.ocr_only_long_caption and has_long_caption and self.ocr_in_caption(long_caption):
            long_short_caption = [long_caption]
            # if random.random() < 0.1:
            #     self.logger.info(f"ocr_only_long_caption: {self.ocr_only_long_caption}, has_long_caption: {has_long_caption}, long_caption: {long_caption}, long_short_caption: {long_short_caption}")

        elif self.long_short_caption_mutex:
            # from all_captions, select by sample ratio weights
            caption_keys = [k for k in caption_dict.keys() if 'caption' in k]
            caption_weights = {k: self.caption_sample_ratio[k] for k in caption_keys}
            selected_key = random.choices(
                list(caption_weights.keys()),
                weights=list(caption_weights.values())
            )[0]
            long_short_caption = [caption_dict[selected_key]]

            # renorm_probs = [self.caption_sample_ratio["short caption"], self.caption_sample_ratio["long caption"]]
            # renorm_probs = [x * 1.0 / sum(renorm_probs) for x in renorm_probs]
            # if random.random() < renorm_probs[0]:
            #     long_short_caption = [caption_dict["short caption"]]
            # else:
            #     long_short_caption = [caption_dict["long caption"]]

        elif len(long_short_caption) == 0:
            long_short_caption = random.choice([
                [caption_dict["short caption"]],
                [caption_dict["long caption"]],
                [caption_dict["short caption"], caption_dict["long caption"]],
            ])
        # 随机替换关键词顺序
        random.shuffle(tag_list)
        random.shuffle(long_short_caption)

        # 随机替换关键词与文本描述的顺序
        final_string = [", ".join(tag_list), ", ".join(long_short_caption)]
        final_string = [x for x in final_string if len(x)]
        random.shuffle(final_string)
        if hq_source_tag:
            final_string.insert(0, hq_source_tag)
        final_string = ". ".join(final_string).replace(".,", ",").replace(",.", ".").replace("..", ".").replace(",,", ",")
        final_string = re.sub(r'\s+', ' ', final_string).strip()
        return final_string

    def add_ocr_info(self, caption_dict):
        import re
        for key in caption_dict:
            if 'caption' not in key:
                continue
            caption = caption_dict[key]
            ocr_in_caption = re.findall(r'"([^"]*)"', caption)
            if len(ocr_in_caption) <= 0:
                return caption_dict

            ocr_in_caption_dedup = []
            for ocr in ocr_in_caption:
                if f'"{ocr}"' not in ocr_in_caption_dedup:
                    ocr_in_caption_dedup.append(f'"{ocr}"')
            ocr_in_caption = ", ".join(ocr_in_caption_dedup)
            add_ocr_info = f"Characters in the image: {ocr_in_caption}."
            caption_dict[key] = caption_dict[key] + " " + add_ocr_info

        return caption_dict

    def normalize_dict_keys(self, d):
        if d is None:
            return {}
        new_dict = {}
        for key in list(d.keys()):
            if "_" in key:
                new_dict[key.replace("_", " ")] = d[key]
            else:
                new_dict[key] = d[key]
        return new_dict

    def replace_style(self, caption_dict, style_op):
        style_in_caption = caption_dict['style'].strip()
        try:
            style_op_2nd_level = str(style_op).strip().strip(',').split(',')[-1] # str() for nan
            if style_in_caption.lower() in ['realistic photography'] and style_op_2nd_level in ['油画']:
                caption_dict['style'] = random.choice(['oil painting', 'oil painting style'])
        except Exception as e:
            rank0_print(self.logger, f'encounterd parsing error:{e}')
        return caption_dict

    def caption_aug(self, raw_string, style_op=None):
        caption_dict = self.safe_load_string(raw_string)
        caption_dict = self.normalize_dict_keys(caption_dict)
        if style_op is not None:
            self.replace_style(caption_dict, style_op)
        if "short caption" in caption_dict and self.use_ip_tag_map:
            caption_dict["short caption"] = self.random_drop_ip(caption_dict["short caption"])
            '''
            if "source" in caption_dict and caption_dict["source"] in ['t2v_sft_number']:
                for key in caption_dict:
                    if 'caption' in key:
                        caption_dict[key] = caption_dict['short caption']
            '''
        if "long caption" in caption_dict:
            if self.use_ip_tag_map:
                caption_dict["long caption"] = self.random_drop_ip(caption_dict["long caption"])
            if "source" in caption_dict and caption_dict["source"] in ['text_in_video', 't2v_sft_ocr',
                                                                       't2v_sft_ocr_250208', 't2v_sft_ocr_250212']:
                for key in caption_dict:
                    if 'caption' in key:
                        caption_dict[key] = caption_dict['long caption']
            else:
                caption_dict["long caption"] = self.random_drop_word(caption_dict["long caption"])
            # if "source" in caption_dict and caption_dict["source"] == 't2v_sft_ocr_250208':
            # caption_dict = self.add_ocr_info(caption_dict)
            # print("================================================")
            # print(f"caption_dict: {caption_dict}")
            # print("================================================")

            # if "source" in caption_dict and caption_dict["source"] == 't2v_sft_ocr_250208':
            #     for key in caption_dict:
            #         if 'caption' in key:
            #             caption_dict[key] = caption_dict['long caption']
            caption = self.random_compose(caption_dict)
        return caption
