def get_body_shape(bust, waist, hip):
    if (bust < 60 or hip < 60 or bust > 120 or hip > 120 or waist < 50 or waist > 120):
        return 'Invalid measurements'
        
    try:
        bust = int(bust/2.54)
        waist = int(waist/2.54)
        hip = int(hip/2.54)

        if float(waist) * float(1.2) <= bust & hip:
            body_shape = 'Hourglass'

        elif float(hip) * float(1.05) > bust:
            body_shape = 'Pear'

        elif float(hip) * float(1.05) < bust:
            body_shape = 'Strawberry'
            
        elif float(waist) > bust & hip:
            body_shape = 'Apple'

        high = max(bust, waist, hip)
        low = min(bust, waist, hip)
        difference = high - low
        # print(high, low, difference)

        if difference <= 5:
            body_shape = 'Banana'
            
        return body_shape
    except ValueError:
        print('Invalid measurements')
    
if __name__ == '__main__':
    print(get_body_shape(90, 80, 90))