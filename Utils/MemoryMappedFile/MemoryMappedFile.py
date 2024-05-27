import mmap


class MemoryMappedFileHandler:
    def __init__(self, file_name, video_width, video_height):
        self.file_name = file_name
        self.video_width = video_width
        self.video_height = video_height
        self.size = int(self.video_width * self.video_height * 3)
        self.mapped_file = self.open_file()

    def open_file(self):
        return mmap.mmap(-1, self.size, access=mmap.ACCESS_READ, tagname=self.file_name)

    def read_content(self):
        content = self.mapped_file.read()
        self.mapped_file.seek(0)
        return content

    def close_file(self):
        self.mapped_file.close()
