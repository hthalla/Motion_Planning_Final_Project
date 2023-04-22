import math
def discr_cor(safe_confs, cell_size=0.5):
    sf_x = safe_confs[0]
    sf_y = safe_confs[1]

    ds_x = math.ceil(sf_x / cell_size) - 1
    ds_y = math.ceil(sf_y / cell_size) - 1
    
    if sf_x % cell_size == 0:
        ds_x += 1
    if sf_y % cell_size == 0:
        ds_y += 1
        
    return (ds_x, ds_y)

x,y = discr_cor((0.6,1.1))

print(x)
print(y)