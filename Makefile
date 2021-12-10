camera_calibrate:
	g++ camera_calibrate.cpp -o camera_calibrate

.PHONY: clean
clean:
	rm -rf camera_calibrate