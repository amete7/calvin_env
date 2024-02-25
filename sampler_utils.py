import numpy as np

def get_task_label(lang,task_oracle):
    lang = lang.lower()
    lang = lang.replace(",", "")
    lang = lang.split()
    if "block" in lang or "object" in lang or "blocks" in lang or "objects" in lang:
        if "red" in lang:
            if "rotate" in lang or "turn" in lang:
                if "left" in lang:
                    return task_oracle.task_to_id["rotate_red_block_left"]
                elif "right" in lang:
                    return task_oracle.task_to_id["rotate_red_block_right"]
            elif "push" in lang or "slide" in lang:
                if "left" in lang:
                    return task_oracle.task_to_id["push_red_block_left"]
                elif "right" in lang:
                    return task_oracle.task_to_id["push_red_block_right"]
            elif "lift" in lang or "take" in lang or "grasp" in lang or "pick" in lang:
                if "table" in lang:
                    return task_oracle.task_to_id["lift_red_block_table"]
                elif "cabinet" in lang or "slider" in lang:
                    return task_oracle.task_to_id["lift_red_block_slider"]
                elif "drawer" in lang:
                    return task_oracle.task_to_id["lift_red_block_drawer"]
                else:
                    return task_oracle.task_to_id["lift_red_block_table"]
            elif "place" in lang or "put" in lang:
                if "cabinet" in lang or "slider" in lang:
                    return task_oracle.task_to_id["place_in_slider"]
                elif "drawer" in lang:
                    return task_oracle.task_to_id["place_in_drawer"]
                elif "another" in lang or "top" in lang:
                    return task_oracle.task_to_id["stack_block"]
            elif "slide" in lang or "sweep" in lang or "push" in lang and "drawer" in lang:
                return task_oracle.task_to_id["push_into_drawer"]

        elif "blue" in lang:
            if "rotate" in lang or "turn" in lang:
                if "left" in lang:
                    return task_oracle.task_to_id["rotate_blue_block_left"]
                elif "right" in lang:
                    return task_oracle.task_to_id["rotate_blue_block_right"]
            elif "push" in lang:
                if "left" in lang:
                    return task_oracle.task_to_id["push_blue_block_left"]
                elif "right" in lang:
                    return task_oracle.task_to_id["push_blue_block_right"]
            elif "lift" in lang or "take" in lang or "grasp" in lang or "pick" in lang:
                if "table" in lang:
                    return task_oracle.task_to_id["lift_blue_block_table"]
                elif "cabinet" in lang or "slider" in lang:
                    return task_oracle.task_to_id["lift_blue_block_slider"]
                elif "drawer" in lang:
                    return task_oracle.task_to_id["lift_blue_block_drawer"]
                else:
                    return task_oracle.task_to_id["lift_blue_block_table"]
            elif "place" in lang or "put" in lang:
                if "cabinet" in lang or "slider" in lang:
                    return task_oracle.task_to_id["place_in_slider"]
                elif "drawer" in lang:
                    return task_oracle.task_to_id["place_in_drawer"]
                elif "another" in lang or "top" in lang:
                    return task_oracle.task_to_id["stack_block"]
            elif "slide" in lang or "sweep" in lang or "push" in lang and "drawer" in lang:
                return task_oracle.task_to_id["push_into_drawer"]

        elif "pink" in lang:
            if "rotate" in lang or "turn" in lang:
                if "left" in lang:
                    return task_oracle.task_to_id["rotate_pink_block_left"]
                elif "right" in lang:
                    return task_oracle.task_to_id["rotate_pink_block_right"]
            elif "push" in lang:
                if "left" in lang:
                    return task_oracle.task_to_id["push_pink_block_left"]
                elif "right" in lang:
                    return task_oracle.task_to_id["push_pink_block_right"]
            elif "lift" in lang or "take" in lang or "grasp" in lang or "pick" in lang:
                if "table" in lang:
                    return task_oracle.task_to_id["lift_pink_block_table"]
                elif "cabinet" in lang or "slider" in lang:
                    return task_oracle.task_to_id["lift_pink_block_slider"]
                elif "drawer" in lang:
                    return task_oracle.task_to_id["lift_pink_block_drawer"]
                else:
                    return task_oracle.task_to_id["lift_pink_block_table"]
            elif "place" in lang or "put" in lang:
                if "cabinet" in lang or "slider" in lang:
                    return task_oracle.task_to_id["place_in_slider"]
                elif "drawer" in lang:
                    return task_oracle.task_to_id["place_in_drawer"]
                elif "another" in lang or "top" in lang:
                    return task_oracle.task_to_id["stack_block"]
            elif "slide" in lang or "sweep" in lang or "push" in lang and "drawer" in lang:
                return task_oracle.task_to_id["push_into_drawer"]
        
        elif "place" in lang or "put" in lang or "store" in lang:
                if "cabinet" in lang or "slider" in lang:
                    return task_oracle.task_to_id["place_in_slider"]
                elif "drawer" in lang:
                    return task_oracle.task_to_id["place_in_drawer"]
                elif "another" in lang or "top" in lang:
                    return task_oracle.task_to_id["stack_block"]
        elif "slide" in lang or "sweep" in lang or "push" in lang and "drawer" in lang:
                return task_oracle.task_to_id["push_into_drawer"]
        elif "unstack" in lang or "remove" in lang or "collapse" in lang or "off" in lang:
            return task_oracle.task_to_id["unstack_block"]
        elif "stack" in lang:
            if "remove" in lang or "collapse" in lang:
                return task_oracle.task_to_id["unstack_block"]
            else:
                return task_oracle.task_to_id["stack_block"]
    
    elif "place" in lang or "put" in lang:
        if "cabinet" in lang or "slider" in lang:
            return task_oracle.task_to_id["place_in_slider"]
        elif "drawer" in lang:
            return task_oracle.task_to_id["place_in_drawer"]
        elif "another" in lang or "top" in lang:
            return task_oracle.task_to_id["stack_blocks"]
    elif "into" in lang and "drawer" in lang:
            return task_oracle.task_to_id["push_into_drawer"]

    elif "drawer" in lang:
        if "pull" in lang or "open" in lang:
            return task_oracle.task_to_id["open_drawer"]
        elif "push" in lang or "close" in lang:
            return task_oracle.task_to_id["close_drawer"]
    elif "slider" in lang or "cabinet" in lang or "door" in lang:
        if "right" in lang:
            return task_oracle.task_to_id["move_slider_right"]
        elif "left" in lang:
            return task_oracle.task_to_id["move_slider_left"]
    elif "switch" in lang:
        if "up" in lang or "upwards" in lang or "on" in lang:
            return task_oracle.task_to_id["turn_on_lightbulb"]
        elif "down" in lang or "downwards" in lang or "off" in lang:
            return task_oracle.task_to_id["turn_off_lightbulb"]
    elif "bulb" in lang or "yellow" in lang:
        if "on" in lang:
            return task_oracle.task_to_id["turn_on_lightbulb"]
        elif "off" in lang:
            return task_oracle.task_to_id["turn_off_lightbulb"]
    elif "button" in lang:
        if "on" in lang:
            return task_oracle.task_to_id["turn_on_led"]
        elif "off" in lang:
            return task_oracle.task_to_id["turn_off_led"]
    elif "led" in lang or "green" in lang:
        if "on" in lang:
            return task_oracle.task_to_id["turn_on_led"]
        elif "off" in lang:
            return task_oracle.task_to_id["turn_off_led"]
    lang = " ".join(lang)
    raise ValueError(f"Task not found for language: {lang}")

if __name__ == '__main__':
    get_task_label()