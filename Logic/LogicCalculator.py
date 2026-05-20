from mahjong.shanten import Shanten
# from mahjong.hand_calculating.hand import HandCalculator # To be used later for Yaku

class LogicCalculator:
    def __init__(self):
        self.shanten_calc = Shanten()
        # self.hand_calc = HandCalculator()
        
    def calculate_shanten(self, hand_34_array):
        """
        Returns the tiles away from tenpai.
        """
        return self.shanten_calc.calculate_shanten(hand_34_array)
        
    def calculate_best_discards(self, hand_34_array, discarded_tiles):
        """
        Calculates the best possible discards to reduce the shanten.
        Uses Ukeire (tile acceptance) to rank the best discards.
        Returns the minimum shanten and a list of tuples (discard_tile, ukeire_count) 
        sorted by the highest number of accepted tiles.
        """
        min_shanten = 99
        discard_ukeire_map = {}
        
        # 1. Simulate Discards: Loop through all tiles we currently have
        for discard_tile in range(34):
            if hand_34_array[discard_tile] > 0:
                # Simulate the discard
                hand_34_array[discard_tile] -= 1
                
                current_shanten = self.shanten_calc.calculate_shanten(hand_34_array)
                
                # Check if it improves or matches our minimum found shanten
                if current_shanten < min_shanten:
                    min_shanten = current_shanten
                    discard_ukeire_map.clear()
                
                if current_shanten == min_shanten:
                    ukeire_count = 0
                    
                    # 2. Simulate Draws: For the new 13-tile hand, try all 34 tiles
                    for draw_tile in range(34):
                        if hand_34_array[draw_tile] < 4:
                            # Temporarily draw
                            hand_34_array[draw_tile] += 1
                            new_shanten = self.shanten_calc.calculate_shanten(hand_34_array)
                            
                            # 3. Check Shanten: If it dropped, this is an accepted tile (Ukeire)
                            if new_shanten < current_shanten:
                                # 4. Count: Add the remaining tiles of this type
                                # (Since we already added 1 to hand_34_array, available is 5 - count)
                                ukeire_count += 5 - hand_34_array[draw_tile] - discarded_tiles[draw_tile]
                                
                            # Revert the draw
                            hand_34_array[draw_tile] -= 1
                            
                    discard_ukeire_map[discard_tile] = ukeire_count
                    
                # Revert the discard simulation
                hand_34_array[discard_tile] += 1
                
        # Rank the discards by ukeire count descending
        ranked_discards = sorted(discard_ukeire_map.items(), key=lambda item: item[1], reverse=True)
                
        return min_shanten, ranked_discards
