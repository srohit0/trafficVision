#
#
# Copyright (c) 2018. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in

class estimate:
    def __init__(self):
        self.speed_limit         = 35.0; # miles/hr
        self.fps                 = 30
        self.y_range             = (160,232)
        #self.y_range             = (194,200)
        #self.y_range             = (0,1080)
        self.caliberation_factor = 10.0; # specific to camera angle/distance
        self.scale_factors       = []; # to compute speed beyond y-range.
        self.last_boxes          = []


    # caliberated for  media/VID_20180709_111331.mp4
    def mph(self, vehicle_center, pix_dist, vdir):
        expected_dist = pix_dist;
        if ( vdir == "down" ):
            expected_dist = 0.25*(vehicle_center-400)
        if ( vdir == "up" ):
            expected_dist = -0.18*(vehicle_center-580)

        return int(self.speed_limit * (pix_dist/expected_dist))

    def color(self, speed):
        red   = 0
        green = 255
        if ( speed > 2*self.speed_limit ):
            red   = 255;
            green = 0;
        elif ( speed > self.speed_limit ):
            red   = 255
            green = 255-int(255*((speed-self.speed_limit)/self.speed_limit))
        return (0, green, red)


    def speed(self, boxes):
        # calibration video = media/VID_20180709_111331.mp4

        if ( len(self.last_boxes) == 0 ):
            self.last_boxes = boxes

        new_boxes = []
        for box, last_box in zip(boxes, self.last_boxes):
            box_speed = box
            box_speed.append(-1.0); # unknown speed
            box_speed.append("unknown"); # unknown direction
            box_center_y = (box[1] + box[3])/2
            if ( last_box != box and
                  ((box_center_y > self.y_range[0] and box_center_y < self.y_range[1]) or
                   (box_center_y > self.y_range[0] and box_center_y < self.y_range[1]) )
               ):

                if ( box[3] > last_box[3] ):
                    pix_dir = "down"
                else:
                    pix_dir = "up"

                pix_distance = last_box[3]-box[3]

                box_speed[7] = self.mph(box_center_y, pix_distance, pix_dir);
                box_speed[8] = pix_dir

            new_boxes.append(box_speed)

        last_boxes = new_boxes
        return new_boxes

