

![Screenshot 2025-06-15 at 1 55 27 AM](https://github.com/user-attachments/assets/27cc75f0-0cee-4ff7-a8a3-a87c63409c50)

# Trade Flow Matrix

โปรเจกต์นี้เป็นเครื่องมือสำหรับวิเคราะห์ Order Flow และแสดงผล Trade Flow Matrix ของคู่เหรียญคริปโต เช่น BTC/USDT โดยใช้ข้อมูลจาก Binance API และแสดงผลผ่าน Streamlit

## คุณสมบัติ
- แสดงกราฟ Trade Flow Matrix วิเคราะห์การเคลื่อนไหวของราคาและปริมาณการซื้อขาย
- ดึงข้อมูลแบบเรียลไทม์จาก Binance
- ตั้งค่าได้ผ่านไฟล์ `config.ini`
- รองรับการรีเฟรชอัตโนมัติ

## วิธีติดตั้ง
1. ติดตั้ง Python 3.7 ขึ้นไป
2. ติดตั้งไลบรารีที่จำเป็น:

   ```bash
   pip install -r requirements.txt
   ```

## การตั้งค่า config.ini
สร้างไฟล์ `config.ini` ในโฟลเดอร์ `trade_flow_matrix` (ตัวอย่างไฟล์):

```ini
[BINANCE]
API_KEY = ใส่_API_KEY_ของคุณ
API_SECRET = ใส่_API_SECRET_ของคุณ

[TRADING]
symbol = BTC/USDT
timeframe = 1m
limit = 1000
```

- `API_KEY` และ `API_SECRET` คือคีย์จาก Binance (สมัครและสร้างได้ที่ [Binance API Management](https://www.binance.com/en/my/settings/api-management))
- `symbol` คือคู่เหรียญที่ต้องการวิเคราะห์ เช่น BTC/USDT
- `timeframe` คือช่วงเวลาแท่งเทียน (เช่น 1m, 5m)
- `limit` คือจำนวนข้อมูล trade ที่จะดึง (แนะนำ 1000 ขึ้นไปเพื่อความลื่นไหล)

## วิธีใช้งาน
1. เปิดเทอร์มินัลและเข้าไปที่โฟลเดอร์ `trade_flow_matrix`
2. รันแอปด้วยคำสั่ง:

   ```bash
   streamlit run trade_flow.py
   ```

3. หน้าเว็บจะเปิดอัตโนมัติ แสดงกราฟ Trade Flow Matrix แบบอินเตอร์แอคทีฟ
4. สามารถตั้งค่า Auto-refresh และช่วงเวลารีเฟรชได้ที่แถบด้านข้าง

## หมายเหตุ
- หากต้องการเปลี่ยนคู่เหรียญหรือจำนวนข้อมูล ให้แก้ไขในไฟล์ `config.ini`
- หากพบปัญหาเกี่ยวกับ API Key หรือการเชื่อมต่อ กรุณาตรวจสอบคีย์และอินเทอร์เน็ต

---

**PopQuants**
