"""NFC Forum URI NDEF record and Type 2 tag TLV (no synthetic audio encoding)."""
from pathlib import Path
def encode_uri(url):
    if not url.startswith(('http://','https://')):raise ValueError('NFC 地址必须是 HTTP 或 HTTPS 链接。')
    prefix=4 if url.startswith('https://') else 3
    payload=bytes([prefix])+url.split('://',1)[1].encode('utf-8')
    short=len(payload)<256
    return bytes([0xD1 if short else 0xC1,1])+(bytes([len(payload)]) if short else len(payload).to_bytes(4,'big'))+b'U'+payload
def write_nfc(folder,url):
    folder=Path(folder);record=encode_uri(url)
    length=bytes([len(record)]) if len(record)<255 else b'\xff'+len(record).to_bytes(2,'big')
    ndef=folder/'memory.ndef';tlv=folder/'memory-nfc-tlv.bin'
    ndef.write_bytes(record);tlv.write_bytes(b'\x03'+length+record+b'\xfe')
    (folder/'nfc-address.txt').write_text(url,encoding='utf-8')
    return str(ndef),str(tlv)
