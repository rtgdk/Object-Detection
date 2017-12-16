import timeit
dataset = [
    "obama-240p.jpg",
    "obama-480p.jpg",
    "obama-720p.jpg",
    "obama-1080p.jpg"
]


def checkBench(setup, testing, itera=5, testToRun=10):
    fast_exe = min(timeit.Timer(testing, setup=setup).repeat(testToRun, itera))
    exeTime = fast_exe / itera
    fps = 1.0 / exeTime
    return exeTime, fps


setLocFace = """
import ceeri_lop
image = ceeri_lop.loadImageFile("{}")
"""

testing_locate_faces = """
facepos = ceeri_lop.facepos(image)
"""

setup_f_mark = """
import ceeri_lop
image = ceeri_lop.loadImageFile("{}")
facepos = ceeri_lop.facepos(image)
"""

testing_f_mark = """
landmarks = ceeri_lop.f_mark(image, face_locations=facepos)[0]
"""

setup_encode_face = """
import ceeri_lop
image = ceeri_lop.loadImageFile("{}")
facepos = ceeri_lop.facepos(image)
"""

testing_encode_face = """
encoding = ceeri_lop.f_ecs(image, gen_f_pos=facepos)[0]
"""

setup_end_to_end = """
import ceeri_lop
image = ceeri_lop.loadImageFile("{}")
"""

testing_end_to_end = """
encoding = ceeri_lop.f_ecs(image)[0]
"""

print("Benchmarks (Note: All benchmarks are only using a single CPU core)")
print()

for image in dataset:
    size = image.split("-")[1].split(".")[0]
    print("Timings at {}:".format(size))

    print(" - Face locations: {:.4f}s ({:.2f} fps)".format(*checkBench(setLocFace.format(image), testing_locate_faces)))
    print(" - Face landmarks: {:.4f}s ({:.2f} fps)".format(*checkBench(setup_f_mark.format(image), testing_f_mark)))
    print(" - Encode face (inc. landmarks): {:.4f}s ({:.2f} fps)".format(*checkBench(setup_encode_face.format(image), testing_encode_face)))
    print(" - End-to-end: {:.4f}s ({:.2f} fps)".format(*checkBench(setup_end_to_end.format(image), testing_end_to_end)))
    print()
