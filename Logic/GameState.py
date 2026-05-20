from matplotlib.pyplot import box


class GameState:
    # A mapping from the 34-array index to the actual tile asset filenames
    TILE_INDEX_TO_ASSET = {
        # Man (0-8)
        0: "Man1.png", 1: "Man2.png", 2: "Man3.png", 3: "Man4.png", 4: "Man5.png",
        5: "Man6.png", 6: "Man7.png", 7: "Man8.png", 8: "Man9.png", 
        # Pin (9-17)
        9: "Pin1.png", 10: "Pin2.png", 11: "Pin3.png", 12: "Pin4.png", 13: "Pin5.png",
        14: "Pin6.png", 15: "Pin7.png", 16: "Pin8.png", 17: "Pin9.png", 
        # Sou (18-26)
        18: "Sou1.png", 19: "Sou2.png", 20: "Sou3.png", 21: "Sou4.png", 22: "Sou5.png",
        23: "Sou6.png", 24: "Sou7.png", 25: "Sou8.png", 26: "Sou9.png",
        # Honors (27-33)
        27: "Ton.png",     # East
        28: "Nan.png",     # South
        29: "Shaa.png",    # West
        30: "Pei.png",     # North
        31: "Haku.png",    # White
        32: "Hatsu.png",   # Green
        33: "Chun.png"     # Red
    }

    SUIT_TO_OFFSET = {
        "m": 0,
        "p": 9,
        "s": 18,
        "z": 27,
    }

    def __init__(self):
        # A 34-length array where the index corresponds to a tile type, and the value is the count.
        # 0-8: Man, 9-17: Pin, 18-26: Sou, 27-33: Honors (East, South, West, North, Haku, Hatsu, Chun)
        self.hand_array = [0] * 34
        self.discarded_tiles = [0] * 34

    @classmethod
    def class_to_idx(cls, class_name):
        suit = class_name[-1] # 'm', 'p', 's', 'z'
        rank = int(class_name[0]) # 0-9
        
        # Convert normal 1-9 to index 0-8 for m,p,s
        if rank == 0:
            rank = 5 # Red fives count as 5
        
        idx = rank - 1 + cls.SUIT_TO_OFFSET[suit]
        return idx

    @classmethod
    def build_hand_array_from_string(cls, hand_string):
        """
        Parse a compact hand string like 12230789m78p3468s into a 34-tile array.
        """
        hand_array = [0] * 34
        if not hand_string:
            return hand_array

        digits = ""

        for char in hand_string.strip().lower():
            if char.isdigit():
                digits += char
                continue

            if char not in cls.SUIT_TO_OFFSET:
                raise ValueError(f"Invalid suit '{char}' in hand string")

            if not digits:
                raise ValueError(f"Missing tiles before suit '{char}'")

            offset = cls.SUIT_TO_OFFSET[char]
            for digit in digits:
                rank = 5 if digit == "0" else int(digit)
                if char == "z":
                    if rank < 1 or rank > 7:
                        raise ValueError("Honor tiles must use ranks 1-7")
                    index = offset + rank - 1
                else:
                    if rank < 1 or rank > 9:
                        raise ValueError("Suit tiles must use ranks 1-9")
                    index = offset + rank - 1

                if hand_array[index] >= 4:
                    raise ValueError("A tile cannot appear more than four times")
                hand_array[index] += 1

            digits = ""

        if digits:
            raise ValueError("Hand string must end with a suit character")

        return hand_array

    def update_from_vision(self, boxes, yolo_names, roi, discard_tiles):
        """
        Translates raw YOLO bounding box detections into structured game domains
        by looking at the spatial layout.
        """
        self.hand_array = [0] * 34
        self.discarded_tiles = [0] * 34

        if boxes is None:
            return
            
        cls_array = boxes.cls.cpu().numpy()
        for i in range(len(cls_array)):
            box = boxes[i]
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            b_cx = (bx1 + bx2) / 2
            b_cy = (by1 + by2) / 2
            if roi[0] <= b_cx <= roi[2] and roi[1] <= b_cy <= roi[3]:
                class_id = int(cls_array[i])
                class_name = yolo_names.get(class_id)
                if not class_name:
                    continue
                    
                idx = self.class_to_idx(class_name)
                    
                if 0 <= idx < 34 and self.hand_array[idx] < 4:
                    self.hand_array[idx] += 1
        
        for tile in discard_tiles:
            idx = self.class_to_idx(tile)
            if 0 <= idx < 34:
                self.discarded_tiles[idx] += 1


    def print_discarded_tiles(self):
        discarded_list = []
        for idx, count in enumerate(self.discarded_tiles):
            if count > 0:
                tile_name = self.TILE_INDEX_TO_ASSET.get(idx, f"Unknown({idx})")
                discarded_list.append(f"{tile_name} x{count}")
        print("Discarded Tiles:", ", ".join(discarded_list) if discarded_list else "None")
        
