.PHONY: clean test run

run:
	mkdir performance 
	cmake -S . -B build
	cmake --build build
	cd build && make

test: test_BGR2GRAY test_CameraCalibrate
	
test_BGR2GRAY:	
	python3 -m pytest test/test_rgb2gray.py -v

test_CameraCalibrate:	
	python3 -m pytest test/test_cameracalibrate.py -v

clean:
	rm -rf .vscode test/__pycache__ ./build ./.pytest_cache ./performance __pycache__