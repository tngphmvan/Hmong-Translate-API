"""
Script test API dịch Mông - Việt
"""
import requests
import sys
import os

API_BASE_URL = "http://localhost:8000"


def test_health():
    """Kiểm tra API có hoạt động không"""
    print("🏥 Kiểm tra health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API đang hoạt động")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ API không phản hồi (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Không thể kết nối đến API: {e}")
        return False


def test_api_info():
    """Lấy thông tin API"""
    print("\nℹ️  Lấy thông tin API...")
    try:
        response = requests.get(API_BASE_URL)
        if response.status_code == 200:
            info = response.json()
            print("✅ Thông tin API:")
            print(f"   App: {info.get('app')}")
            print(f"   Version: {info.get('version')}")
            print(f"   Status: {info.get('status')}")
            print(f"   Models: {info.get('models')}")
            return True
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def test_hmong_to_vietnamese(audio_file):
    """Test API Mông -> Việt"""
    print(f"\n🎤 Test API: Mông → Việt")
    print(f"   File: {audio_file}")

    if not os.path.exists(audio_file):
        print(f"❌ File không tồn tại: {audio_file}")
        return False

    try:
        with open(audio_file, "rb") as f:
            files = {"audio": f}
            print("   Đang gửi request...")
            response = requests.post(
                f"{API_BASE_URL}/api/hmong-to-vietnamese",
                files=files,
                timeout=120  # 2 phút timeout
            )

        if response.status_code == 200:
            result = response.json()
            print("✅ Dịch thành công!")
            print(f"   Tiếng Mông: {result['hmong_text']}")
            print(f"   Tiếng Việt: {result['vietnamese_text']}")
            return True
        else:
            print(f"❌ Lỗi: {response.status_code}")
            print(f"   Detail: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def test_vietnamese_to_hmong(audio_file, output_file="output_hmong.wav"):
    """Test API Việt -> Mông"""
    print(f"\n🎤 Test API: Việt → Mông")
    print(f"   Input: {audio_file}")
    print(f"   Output: {output_file}")

    if not os.path.exists(audio_file):
        print(f"❌ File không tồn tại: {audio_file}")
        return False

    try:
        with open(audio_file, "rb") as f:
            files = {"audio": f}
            print("   Đang gửi request...")
            response = requests.post(
                f"{API_BASE_URL}/api/vietnamese-to-hmong",
                files=files,
                timeout=120  # 2 phút timeout
            )

        if response.status_code == 200:
            # Lưu file audio
            with open(output_file, "wb") as out:
                out.write(response.content)

            # Lấy thông tin từ headers
            vi_text = response.headers.get('X-Vietnamese-Text', 'N/A')
            hmong_text = response.headers.get('X-Hmong-Text', 'N/A')

            print("✅ Dịch thành công!")
            print(f"   Tiếng Việt: {vi_text}")
            print(f"   Tiếng Mông: {hmong_text}")
            print(f"   File đã lưu: {output_file}")
            return True
        else:
            print(f"❌ Lỗi: {response.status_code}")
            print(f"   Detail: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def main():
    print("=" * 60)
    print("Test Hmong-Vietnamese Translation API")
    print("=" * 60)

    # Test 1: Health check
    if not test_health():
        print("\n⚠️  API chưa khởi động. Vui lòng chạy: python api.py")
        return

    # Test 2: API info
    test_api_info()

    # Test 3: Mông -> Việt
    if len(sys.argv) > 1:
        hmong_audio = sys.argv[1]
        test_hmong_to_vietnamese(hmong_audio)
    else:
        print("\n⚠️  Bỏ qua test Mông → Việt (không có file audio)")
        print("   Sử dụng: python test_api.py <hmong_audio.wav>")

    # Test 4: Việt -> Mông
    if len(sys.argv) > 2:
        vietnamese_audio = sys.argv[2]
        test_vietnamese_to_hmong(vietnamese_audio)
    else:
        print("\n⚠️  Bỏ qua test Việt → Mông (không có file audio)")
        print("   Sử dụng: python test_api.py <hmong_audio.wav> <vietnamese_audio.wav>")

    print("\n" + "=" * 60)
    print("Hoàn thành test!")
    print("=" * 60)


if __name__ == "__main__":
    main()
