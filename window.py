from main import process_image, process_video
import tkinter.filedialog
import threading
import tkinter
import ctypes
import pygame
import time
import os
pygame.init()


def get_color(c1, c2, n):
    """颜色变换"""
    def limit(num):
        return num if 0 <= num <= 255 else (0 if num < 0 else 255)
    return limit(n(c1[0], c2[0])), limit(n(c1[1], c2[1])), limit(n(c1[2], c2[2]))


def find_window(win_cls, win_name):
    """找窗口句柄"""
    h = ctypes.windll.user32.FindWindowA(win_cls, win_name)
    if not h:
        h = ctypes.windll.user32.FindWindowW(win_cls, win_name)
    return h


class Widget:
    """组件"""
    window = None

    @classmethod
    def init(cls, win):
        cls.window = win

    def __init__(self, x, y, width, height, color=(200, 200, 200)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.surface = pygame.Surface((width, height))
        self.surface.fill(color)
        if self.window:
            self.window.add(self)

    def draw(self, surface):
        surface.blit(self.surface, (self.x, self.y))

    def mouse_down(self):
        pass

    def mouse_up(self):
        pass

    def mouse_move(self, pos):
        pass


class Button(Widget):
    """按钮"""

    def __init__(self, x, y, width, height, text, color=(200, 200, 200), hovered_color='auto', downed_color='auto',
                 font_color=(0, 0, 0), size=20, callback=None, args=(), kwargs=None):
        super(Button, self).__init__(x, y, width, height, color)
        self.text = text
        self.text_surface = pygame.font.SysFont('console', size).render(text,  True, font_color)

        if hovered_color == 'auto' and downed_color != 'auto':
            def n_mid(n1, n2):
                return n1 + (n2 - n1) // 2
            hovered_color = get_color(color, downed_color, n_mid)
        elif downed_color == 'auto' and hovered_color != 'auto':
            def n_side(n1, n2):
                return n2 + (n2 - n1)
            downed_color = get_color(color, hovered_color, n_side)
        elif downed_color == hovered_color == 'auto':
            def sub(c):
                return c[0] - m0, c[1] - m1, c[2] - m2
            m0, m1, m2 = [40 if color[i] > 80 else color[i] * 0.4 for i in range(3)]
            hovered_color = sub(color)
            downed_color = sub(hovered_color)
        self.bg_color = color
        self.hovered_color = hovered_color
        self.downed_color = downed_color
        self.border_color = (0, 0, 0)

        self.callback = callback
        self.args = args
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs

        self.downed = False
        self.hovered = False
        self.render()

    def in_rect(self, x, y):
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def render(self):
        font_x = (self.width - self.text_surface.get_width()) // 2
        font_y = (self.height - self.text_surface.get_height()) // 2
        self.surface.fill(self.color)
        self.surface.blit(self.text_surface, (font_x, font_y))
        pygame.draw.rect(self.surface, self.border_color, self.surface.get_rect(), 1)

    def mouse_move(self, pos):
        if self.in_rect(*pos):
            if not self.hovered:
                self.on_hover()
                self.hovered = True
        elif self.hovered:
            self.on_lost_hover()
            self.hovered = False

    def mouse_down(self):
        if self.hovered and not self.downed:
            self.on_mouse_down()
            self.downed = True

    def mouse_up(self):
        self.on_mouse_up()
        if self.downed:
            if self.hovered:
                self.on_click()
            self.downed = False

    def on_mouse_down(self):
        self.color = self.downed_color
        self.render()

    def on_mouse_up(self):
        if self.hovered:
            self.color = self.hovered_color
        else:
            self.color = self.bg_color
        self.render()

    def on_click(self):
        if self.callback:
            self.callback(*self.args, **self.kwargs)

    def on_hover(self):
        self.color = self.hovered_color
        self.render()

    def on_lost_hover(self):
        self.color = self.bg_color
        self.render()


class Label(Widget):
    """文本"""

    def __init__(self, x, y, width, height, text, color=(200, 200, 200), font_color=(0, 0, 0), size=20):
        super(Label, self).__init__(x, y, width, height, color)
        self.text = text
        self.text_surface = pygame.font.SysFont('console', size).render(text, True, font_color)
        self.render()

    def render(self):
        font_x = (self.width - self.text_surface.get_width()) // 2
        font_y = (self.height - self.text_surface.get_height()) // 2
        self.surface.fill(self.color)
        self.surface.blit(self.text_surface, (font_x, font_y))


class FileView(Widget):
    """文件视图"""

    def __init__(self, x, y, width, height, color=(200, 200, 200), font_color=(0, 0, 0), size=20):
        super(FileView, self).__init__(x, y, width, height, color)
        self.font_color = font_color
        self.size = size
        self.file_lines = []
        self.paths = []

    def draw(self, surface):
        super(FileView, self).draw(surface)
        n = 2
        pygame.draw.rect(surface, self.font_color, (self.x-n, self.y-n, self.width+2*n-1, self.height+2*n-1), n)

    def add(self, path):
        """添加文件"""
        if len(self) < self.height // FileLine.height:
            path = os.path.abspath(path)
            if os.path.isfile(path) and path.split('.')[-1] in ['mp4', 'jpg', 'png'] and path not in self.paths:
                self.file_lines.append(FileLine(self, path))
                self.paths.append(path)

    def remove(self, file_line, flag=False):
        """移除文件"""
        if not self.window.processed or flag:
            index = self.paths.index(file_line.path)
            self.paths.pop(index)
            self.window.children.remove(self.file_lines.pop(index))
            for i in range(index, len(self.file_lines)):
                self.file_lines[i].y -= FileLine.height

    def __len__(self):
        return len(self.file_lines)


class FileLine(Button):
    """文件组件"""
    height = 50

    def __init__(self, file_view, path):
        self.n = 0
        super(FileLine, self).__init__(file_view.x, file_view.y + len(file_view) * FileLine.height,
                                       file_view.width, FileLine.height, os.path.basename(path), file_view.color,
                                       font_color=file_view.font_color, size=file_view.size,
                                       callback=file_view.remove, args=(self, ))
        self.size = file_view.size
        self.font_color = file_view.font_color
        self.path = path

    def render(self):
        font_x = (self.width - self.text_surface.get_width()) // 2
        font_y = (self.height - self.text_surface.get_height()) // 2
        self.surface.fill(self.color)
        if self.n > 0:
            pygame.draw.rect(self.surface, (40, 200, 40), (0, 0, self.width * self.n, self.height))
        self.surface.blit(self.text_surface, (font_x, font_y))

    def update(self, n):
        if n == 1:
            self.hovered_color = self.downed_color = self.bg_color = self.color = (40, 200, 40)
            self.text_surface = pygame.font.SysFont('console', self.size).render(
                self.text + '  Successfully',  True, self.font_color)
        else:
            self.text_surface = pygame.font.SysFont('console', self.size).render(
                self.text + '  Processing...',  True, self.font_color)
        self.n = n
        self.render()


class Window:
    def __init__(self):
        self.tk = tkinter.Tk()
        self.tk.geometry("100x100+300+100")
        title = 'abcdef_tk'
        self.tk.title(title)
        self.tk.update()
        h = find_window(None, title)
        ctypes.windll.user32.ShowWindow(h, 0)
        title = 'Crowd Counting Demo'
        self.screen = pygame.display.set_mode((500, 400))
        pygame.display.set_caption(title)
        self.hwnd = find_window(None, title)
        self.file_view = FileView(40, 60, 240, 300, (255, 255, 255), size=15)
        self.processed = False
        self.children = []
        self.life = True

    def show(self):
        """显示窗口"""
        ctypes.windll.user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 3)
        while self.life:
            for event in pygame.event.get():
                if event.type == pygame.DROPFILE and not self.processed:
                    self.file_view.add(event.file)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for child in self.children:
                        child.mouse_down()
                elif event.type == pygame.MOUSEBUTTONUP:
                    for child in self.children:
                        child.mouse_up()
                elif event.type == pygame.MOUSEMOTION:
                    for child in self.children:
                        child.mouse_move(event.pos)
                elif event.type == pygame.QUIT:
                    self.life = False
                    return
            self.screen.fill((255, 255, 255))
            self.file_view.draw(self.screen)
            for child in self.children:
                child.draw(self.screen)
            pygame.display.flip()

    def add(self, element):
        """添加子组件"""
        self.children.append(element)

    def open_file(self):
        """打开文件"""
        if not self.processed:
            self.file_view.add(tkinter.filedialog.askopenfilename(
                title="请选择打开的文件", filetypes=[('file', '.jpg'), ('file', '.png'), ('file', '.mp4')],
                parent=self.tk))
            ctypes.windll.user32.SetFocus(self.hwnd)

    def process(self):
        """处理输入文件"""
        def process_thread():
            if not os.path.exists('./output'):
                os.mkdir('output')
            if os.path.isdir('./output'):
                for file_line in self.file_view.file_lines:
                    file_type = file_line.path.split('.')[-1]
                    opt_filename = os.path.join('output', 'output_' + os.path.basename(file_line.path))
                    if file_type in ['jpg', 'png']:
                        file_line.update(0)
                        process_image(file_line.path, opt_filename)
                        file_line.update(1)
                    elif file_type in ['mp4']:
                        for n in process_video(file_line.path, opt_filename, 2):
                            file_line.update(n)
                time.sleep(1)
                while self.file_view.file_lines:
                    self.file_view.remove(self.file_view.file_lines[0], True)
                h = find_window(None, 'output')
                if h == 0:
                    os.popen('explorer "output"')
                else:
                    ctypes.windll.user32.ShowWindow(h, 1)
                    ctypes.windll.user32.SetWindowPos(h, -1, 0, 0, 0, 0, 3)
            self.processed = False
        if not self.processed:
            self.processed = True
            t = threading.Thread(target=process_thread)
            t.setDaemon(True)
            t.start()


def init_window():
    """初始化窗口"""
    window = Window()
    Widget.init(window)
    Label(100, 10, 300, 35, "Crowd Counting Demo", color=(255, 255, 255))
    Label(305, 105, 180, 30, "Press the button", size=19, color=(255, 255, 255))
    Label(305, 135, 180, 30, "OR", size=19, color=(255, 255, 255))
    Label(305, 165, 180, 30, "Drop the file", size=19, color=(255, 255, 255))
    Button(320, 220, 150, 50, "Open File", callback=window.open_file)
    Button(320, 300, 150, 50, "Start", callback=window.process)
    return window


def main():
    window = init_window()
    window.show()
    pygame.quit()


if __name__ == '__main__':
    main()
