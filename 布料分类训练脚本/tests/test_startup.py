import socket

from train_cloth import configure_qt_platform


def test_native_wayland_socket_selects_wayland_backend(tmp_path):
    address=tmp_path/'wayland-test'
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(address))
        env={'XDG_SESSION_TYPE':'wayland','XDG_RUNTIME_DIR':str(tmp_path),'WAYLAND_DISPLAY':'wayland-test','DISPLAY':':0'}
        configure_qt_platform(env)
        assert env['QT_QPA_PLATFORM']=='wayland'


def test_explicit_backend_and_screenshot_mode_are_preserved():
    env={'QT_QPA_PLATFORM':'offscreen','XDG_SESSION_TYPE':'wayland'}
    configure_qt_platform(env)
    assert env['QT_QPA_PLATFORM']=='offscreen'
    env={}
    configure_qt_platform(env,screenshot=True)
    assert env['QT_QPA_PLATFORM']=='offscreen'


def test_stale_wayland_environment_does_not_override_x11(tmp_path):
    env={'XDG_SESSION_TYPE':'wayland','XDG_RUNTIME_DIR':str(tmp_path),'WAYLAND_DISPLAY':'missing-socket','DISPLAY':':0'}
    configure_qt_platform(env)
    assert 'QT_QPA_PLATFORM' not in env


def test_gnome_default_socket_is_used_when_display_variable_missing(tmp_path):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(tmp_path/'wayland-0'))
        env={'XDG_SESSION_TYPE':'wayland','XDG_RUNTIME_DIR':str(tmp_path)}
        configure_qt_platform(env)
        assert env['QT_QPA_PLATFORM']=='wayland'
        assert env['WAYLAND_DISPLAY']=='wayland-0'


def test_installer_is_cuda_first_with_explicit_cpu_option():
    from pathlib import Path
    script=(Path(__file__).resolve().parents[1]/'安装依赖.sh').read_text()
    assert '${1:---cuda}' in script and '--cpu)' in script
    assert '+cu130' in script and 'diagnose-gpu' in script
