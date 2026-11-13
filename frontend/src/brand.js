// ============ Thuong hieu: sua o day de doi ten, logo, anh nen ============
// Anh va logo nam trong frontend/public/brand/. Muon doi: chep de tep moi cung ten,
// hoac dat tep ten khac vao thu muc do roi sua duong dan ben duoi. Sau do chay lai `npm run build`.
export const BRAND = {
  name: "Phân loại kỹ thuật cầu lông",
  shortName: "Cầu lông AI",
  logo: "/brand/logo.svg",          // khuyen dung anh vuong (SVG hoac PNG 128x128 tro len)
  loginBackground: "/brand/login-bg.jpg", // anh ngang, toi thieu 1600px chieu rong
  pageBackground: "/brand/login-bg.jpg",  // anh chim cho cac trang con lai (co the dung anh khac)
  pageBackgroundStrength: 0.22,            // do ro cua anh chim: 0 = an han, 1 = ro nhu trang dang nhap
};
